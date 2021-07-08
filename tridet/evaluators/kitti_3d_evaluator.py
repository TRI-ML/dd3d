# Copyright 2021 Toyota Research Institute.  All rights reserved.
import itertools
import json
import math
import os
import warnings
from collections import OrderedDict
from functools import partial

import numpy as np
import pandas as pd
from pyquaternion import Quaternion
from tqdm import tqdm

import numba
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.structures.boxes import BoxMode
from detectron2.utils import comm
from iopath.common.file_io import PathManager
from numba import errors as numba_err

from tridet.evaluators.rotate_iou import d3_box_overlap_kernel, rotate_iou_gpu_eval
from tridet.structures.boxes3d import GenericBoxes3D
from tridet.structures.pose import Pose

warnings.simplefilter('ignore', category=numba_err.NumbaDeprecationWarning)

BBOX3D_PREDICTION_FILE = "bbox3d_predictions.json"
KITTI_SUBMISSION_DIR = "kitti_3d_submission"


class KITTI3DEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        iou_thresholds,
        only_prepare_submission=False,
        output_dir=None,
        distributed=False,
    ):
        dataset_dicts = DatasetCatalog.get(dataset_name)
        metadata = MetadataCatalog.get(dataset_name)
        class_names = metadata.thing_classes
        id_to_name = metadata.contiguous_id_to_name

        self._dataset_dicts = {dikt['file_name']: dikt for dikt in dataset_dicts}
        self._id_to_name = id_to_name
        self._class_names = class_names
        self._iou_thresholds = iou_thresholds
        self._only_prepare_submission = only_prepare_submission
        self._output_dir = output_dir
        self._distributed = distributed

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        # List[Dict], each key'ed by category (str) + vectorized 3D box (10) + 2D box (4) + score (1) + file name (str)
        self._predictions_as_json = []

        self._predictions_kitti_format = []
        self._groundtruth_kitti_format = []

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        for input_per_image, pred_per_image in zip(inputs, outputs):
            pred_classes = pred_per_image['instances'].pred_classes
            pred_boxes = pred_per_image['instances'].pred_boxes.tensor
            pred_boxes3d = pred_per_image['instances'].pred_boxes3d
            # pred_boxes3d = pred_per_image['instances'].pred_box3d_as_vec
            scores = pred_per_image['instances'].scores
            scores_3d = pred_per_image['instances'].scores_3d

            file_name = input_per_image['file_name']
            image_id = input_per_image['image_id']

            # predictions
            predictions_kitti = []
            # for class_id, box3d_as_vec, score, box2d in zip(pred_classes, pred_boxes3d, scores, pred_boxes):
            for class_id, box3d, score_3d, box2d, score in zip(
                pred_classes, pred_boxes3d, scores_3d, pred_boxes, scores
            ):
                # class_name = self._metadata.thing_classes[class_id]
                class_name = self._class_names[class_id]

                box3d_as_vec = box3d.vectorize()[0].cpu().numpy()

                pred = OrderedDict(
                    category_id=int(class_id),  # COCO instances
                    category=class_name,
                    bbox3d=box3d_as_vec.tolist(),
                    # COCO instances uses "XYWH". Aligning with it as much as possible
                    bbox=BoxMode.convert(box2d.tolist(), from_mode=BoxMode.XYXY_ABS, to_mode=BoxMode.XYWH_ABS),
                    score=float(score),
                    score_3d=float(score_3d),
                    file_name=file_name,
                    image_id=image_id  # COCO instances
                )
                self._predictions_as_json.append(pred)

                # prediction in KITTI format.
                W, L, H, x, y, z, rot_y, alpha = convert_3d_box_to_kitti(box3d)
                l, t, r, b = box2d.tolist()
                predictions_kitti.append([
                    class_name, -1, -1, alpha, l, t, r, b, H, W, L, x, y, z, rot_y,
                    float(score_3d)
                ])
            self._predictions_kitti_format.append(pd.DataFrame(predictions_kitti))

            # groundtruths
            gt_dataset_dict = self._dataset_dicts[file_name]

            if "annotations" not in gt_dataset_dict:
                # test set
                continue

            raw_kitti_annotations = gt_dataset_dict.get('raw_kitti_annotations', None)
            if raw_kitti_annotations is not None:
                self._groundtruth_kitti_format.append(raw_kitti_annotations)
            else:
                # Otherwise, use the same format as predictions (minus 'score').
                groundtruth_kitti = []
                for anno in gt_dataset_dict['annotations']:
                    # class_name = self._metadata.thing_classes[anno['category_id']]
                    class_name = self._class_names[anno['category_id']]

                    # groundtruth in KITTI format.
                    box2d = BoxMode.convert(anno['bbox'], from_mode=anno['bbox_mode'], to_mode=BoxMode.XYXY_ABS)
                    box3d = GenericBoxes3D.from_vectors([anno['bbox3d']])
                    W, L, H, x, y, z, rot_y, alpha = convert_3d_box_to_kitti(box3d)
                    l, t, r, b = box2d
                    groundtruth_kitti.append([class_name, -1, -1, alpha, l, t, r, b, H, W, L, x, y, z, rot_y])
                self._groundtruth_kitti_format.append(pd.DataFrame(groundtruth_kitti))

    def evaluate(self):

        if self._distributed:
            comm.synchronize()
            predictions_as_json = comm.gather(self._predictions_as_json, dst=0)
            predictions_as_json = list(itertools.chain(*predictions_as_json))

            predictions_kitti_format = comm.gather(self._predictions_kitti_format, dst=0)
            predictions_kitti_format = list(itertools.chain(*predictions_kitti_format))

            groundtruth_kitti_format = comm.gather(self._groundtruth_kitti_format, dst=0)
            groundtruth_kitti_format = list(itertools.chain(*groundtruth_kitti_format))

            if not comm.is_main_process():
                return
        else:
            predictions_as_json = self._predictions_as_json
            predictions_kitti_format = self._predictions_kitti_format
            groundtruth_kitti_format = self._groundtruth_kitti_format

        # Write prediction file as JSON.
        PathManager().mkdirs(self._output_dir)
        file_path = os.path.join(self._output_dir, BBOX3D_PREDICTION_FILE)
        with open(file_path, 'w') as f:
            json.dump(predictions_as_json, f, indent=4)

        if self._only_prepare_submission:
            self.prepare_kitti3d_submission(
                predictions_kitti_format, submission_dir=os.path.join(self._output_dir, KITTI_SUBMISSION_DIR)
            )
            return {}

        assert len(predictions_kitti_format) == len(groundtruth_kitti_format)
        formatted_predictions = [
            KITTIEvaluationEngine._format(idx, x, True) for idx, x in enumerate(predictions_kitti_format)
        ]
        formatted_groundtruth = [
            KITTIEvaluationEngine._format(idx, x, False) for idx, x in enumerate(groundtruth_kitti_format)
        ]

        engine = KITTIEvaluationEngine(id_to_name=self._id_to_name)
        results = engine.evaluate(formatted_groundtruth, formatted_predictions, overlap_thresholds=self._iou_thresholds)

        results = OrderedDict({k: 100. * v for k, v in results.items()})

        return results

    @staticmethod
    def prepare_kitti3d_submission(predictions_kitti_format, submission_dir):
        assert not os.path.exists(submission_dir)
        os.makedirs(submission_dir)
        for idx, prediction in tqdm(enumerate(predictions_kitti_format)):
            prediction.to_csv(os.path.join(submission_dir, f"{idx:06d}.txt"), sep=" ", header=False, index=False)


def convert_3d_box_to_kitti(box):
    """Convert a single 3D bounding box (GenericBoxes3D) to KITTI convention. i.e. for evaluation. We
    assume the box is in the reference frame of camera_2 (annotations are given in this frame).

    Usage:
        >>> box_camera_2 = pose_02.inverse() * pose_0V * box_velodyne
        >>> kitti_bbox_params = convert_3d_box_to_kitti(box_camera_2)

    Parameters
    ----------
    box: GenericBoxes3D
        Box in camera frame (X-right, Y-down, Z-forward)

    Returns
    -------
    W, L, H, x, y, z, rot_y, alpha: float
        KITTI format bounding box parameters.
    """
    assert len(box) == 1

    quat = Quaternion(*box.quat.cpu().tolist()[0])
    tvec = box.tvec.cpu().numpy()[0]
    sizes = box.size.cpu().numpy()[0]

    # Re-encode into KITTI box convention
    # Translate y up by half of dimension
    tvec += np.array([0., sizes[2] / 2.0, 0])

    inversion = Quaternion(axis=[1, 0, 0], radians=np.pi / 2).inverse
    quat = inversion * quat

    # Construct final pose in KITTI frame (use negative of angle if about positive z)
    if quat.axis[2] > 0:
        kitti_pose = Pose(wxyz=Quaternion(axis=[0, 1, 0], radians=-quat.angle), tvec=tvec)
        rot_y = -quat.angle
    else:
        kitti_pose = Pose(wxyz=Quaternion(axis=[0, 1, 0], radians=quat.angle), tvec=tvec)
        rot_y = quat.angle

    # Construct unit vector pointing in z direction (i.e. [0, 0, 1] direction)
    # The transform this unit vector by pose of car, and drop y component, thus keeping heading direction in BEV (x-z grid)
    v_ = np.float64([[0, 0, 1], [0, 0, 0]])
    v_ = (kitti_pose * v_)[:, ::2]

    # Getting positive theta angle (we define theta as the positive angle between
    # a ray from the origin through the base of the transformed unit vector and the z-axis
    theta = np.arctan2(abs(v_[1, 0]), abs(v_[1, 1]))

    # Depending on whether the base of the transformed unit vector is in the first or
    # second quadrant we add or subtract `theta` from `rot_y` to get alpha, respectively
    alpha = rot_y + theta if v_[1, 0] < 0 else rot_y - theta
    # Bound from [-pi, pi]
    if alpha > np.pi:
        alpha -= 2.0 * np.pi
    elif alpha < -np.pi:
        alpha += 2.0 * np.pi
    alpha = np.around(alpha, decimals=2)  # KITTI precision

    # W, L, H, x, y, z, rot-y, alpha
    return sizes[0], sizes[1], sizes[2], tvec[0], tvec[1], tvec[2], rot_y, alpha


class KITTIEvaluationEngine():

    _DEFAULT_KITTI_LEVEL_TO_PARAMETER = {
        "levels": ("easy", "moderate", "hard"),
        "max_occlusion": (0, 1, 2),
        "max_truncation": (0.15, 0.3, 0.5),
        "min_height": (40, 25, 25)
    }

    def __init__(self, id_to_name, num_shards=50, sample_points=41):
        self.id_to_name = id_to_name
        self.sample_points = sample_points
        self.num_shards = num_shards
        self.filter_data_fn = partial(
            clean_kitti_data, difficulty_level_to_params=self._DEFAULT_KITTI_LEVEL_TO_PARAMETER
        )

    @staticmethod
    def _format(idx, kitti_format, is_prediction):
        if len(kitti_format) == 0:
            annotations = dict(
                id=f'{idx:06d}',
                name=[],
                truncated=np.array([]),
                occluded=np.array([]),
                alpha=np.array([]),
                bbox=np.empty((0, 4)),
                dimensions=np.empty((0, 3)),
                location=np.empty((0, 3)),
                rotation_y=np.array([]),
                score=np.array([])
            )
            return annotations

        data = np.array(kitti_format)
        annotations = dict(
            id=f'{idx:06d}',
            name=data[:, 0],
            truncated=data[:, 1].astype(np.float64),
            occluded=data[:, 2].astype(np.int64),
            alpha=data[:, 3].astype(np.float64),
            bbox=data[:, 4:8].astype(np.float64),
            dimensions=data[:, 8:11][:, [2, 0, 1]].astype(np.float64),
            location=data[:, 11:14].astype(np.float64),
            rotation_y=data[:, 14].astype(np.float64),
        )

        if is_prediction:
            annotations.update({'score': data[:, 15].astype(np.float64)})
        else:
            annotations.update({'score': np.zeros([len(annotations['bbox'])])})
        return annotations

    def get_shards(self, num, num_shards):
        """Shard number into evenly sized parts. `Remaining` values are put into the last shard.

        Parameters
        ----------
        num: int
            Number to shard

        num_shards: int
            Number of shards

        Returns
        -------
        List of length (num_shards or num_shards +1), depending on whether num is perfectly divisible by num_shards
        """
        assert num_shards > 0, "Invalid number of shards"
        num_per_shard = num // num_shards
        remaining_num = num % num_shards
        full_shards = num_shards * (num_per_shard > 0)
        if remaining_num == 0:
            return [num_per_shard] * full_shards
        else:
            return [num_per_shard] * full_shards + [remaining_num]

    def evaluate(self, gt_annos, dt_annos, overlap_thresholds):
        # pr_curves = self.eval_metric(gt_annos, dt_annos, metric, overlap_thresholds)
        gt_annos, dt_annos = self.validate_anno_format(gt_annos, dt_annos)

        box3d_pr_curves = self.eval_metric(gt_annos, dt_annos, 'BOX3D_AP', overlap_thresholds)
        mAP_3d = self.get_mAP(box3d_pr_curves["precision"], box3d_pr_curves["recall"])

        bev_pr_curves = self.eval_metric(gt_annos, dt_annos, 'BEV_AP', overlap_thresholds)
        mAP_bev = self.get_mAP(bev_pr_curves["precision"], bev_pr_curves["recall"])

        results = OrderedDict()
        for class_i, class_name in self.id_to_name.items():
            for diff_i, diff in enumerate(["Easy", "Moderate", "Hard"]):
                for thresh_i, thresh in enumerate(overlap_thresholds):
                    results['kitti_box3d_r40/{}_{}_{}'.format(class_name, diff, thresh)] = \
                        mAP_3d[class_i, diff_i, thresh_i]
        for class_i, class_name in self.id_to_name.items():
            for diff_i, diff in enumerate(["Easy", "Moderate", "Hard"]):
                for thresh_i, thresh in enumerate(overlap_thresholds):
                    results['kitti_bev_r40/{}_{}_{}'.format(class_name, diff, thresh)] = \
                        mAP_bev[class_i, diff_i, thresh_i]
        return results

    def get_mAP(self, precision, recall):
        """ Get mAP from precision.
        Parameters
        ----------
        precision: np.ndarray
            Numpy array of precision curves at different recalls, of shape
            [num_classes, num_difficulties, num_overlap_thresholds,self.sample_points]

        recall: np.ndarray
            Numpy array of recall values corresponding to each precision, of shape
            [num_classes, num_difficulties, num_overlap_thresholds,self.sample_points]

        Returns
        -------
        ap: np.ndarray
            Numpy array of mean AP evaluated at different points along PR curve.
            Shape [num_classes, num_difficulties, num_overlap_thresholds]
        """
        precisions, recall_spacing = self.get_sampled_precision_recall(precision, recall)
        ap = sum(precisions) / len(recall_spacing)
        return ap

    def get_sampled_precision_recall(self, precision, recall):
        """Given an array of precision, recall values, sample evenly along the recall range, and interpolate the precision
        based on AP from section 6 from https://research.mapillary.com/img/publications/MonoDIS.pdf

        Parameters
        ----------
        precision: np.ndarray
            Numpy array of precision curves at different recalls, of shape
            [num_classes, num_difficulties, num_overlap_thresholds, self.sample_points]

        recall: np.ndarray
            Numpy array of recall values corresponding to each precision, of shape
            [num_classes, num_difficulties, num_overlap_thresholds, self.sample_points]

        Returns
            sampled_precision: list of np.ndarrays, of shape (num_classes, num_difficulties, num_overlap_thresholds)
                The maximum precision values corresponding to the sampled recall.
            sampled_recall: list
                Recall values evenly spaced along the recall range.
        """
        # recall_range = self.recall_range
        recall_range = (0.0, 1.0)
        precisions = []
        # Don't count recall at 0
        recall_spacing = [1. / (self.sample_points - 1) * i for i in range(1, self.sample_points)]
        recall_spacing = list(filter(lambda recall: recall_range[0] <= recall <= recall_range[1], recall_spacing))
        for r in recall_spacing:
            precisions_above_recall = (recall >= r) * precision
            precisions.append(precisions_above_recall.max(axis=3))

        return precisions, recall_spacing

    @staticmethod
    def validate_anno_format(gt_annos, dt_annos):
        """Verify that the format/dimensions for the annotations are correct.
        Keys correspond to defintions here:
        https://github.com/NVIDIA/DIGITS/blob/v4.0.0-rc.3/digits/extensions/data/objectDetection/README.md
        """
        necessary_keys = ['name', 'alpha', 'bbox', 'dimensions', 'location', 'rotation_y', 'score']
        for i, (gt_anno, dt_anno) in enumerate(zip(gt_annos, dt_annos)):
            for key in necessary_keys:
                assert key in gt_anno, "{} not present in GT {}".format(key, i)
                assert key in dt_anno, "{} not present in prediction {}".format(key, i)
                if key in ['bbox', 'dimensions', 'location']:
                    # make sure these fields are 2D numpy array
                    assert len(gt_anno[key].shape) == 2, key
                    assert len(dt_anno[key].shape) == 2, key

            for key in ['truncated', 'occluded', 'alpha', 'rotation_y', 'score']:
                if len(gt_anno[key].shape) == 2:
                    gt_anno[key] = np.squeeze(gt_anno[key], axis=0)
                if len(dt_anno[key].shape) == 2:
                    dt_anno[key] = np.squeeze(dt_anno[key], axis=0)
        return gt_annos, dt_annos

    def eval_metric(self, gt_annos, dt_annos, metric, overlap_thresholds):
        assert len(gt_annos) == len(dt_annos), "Must provide a prediction for every ground truth sample"
        num_ground_truths = len(gt_annos)
        shards = self.get_shards(num_ground_truths, self.num_shards)

        overlaps, overlaps_by_shard, total_gt_num, total_dt_num = \
            self.calculate_match_degree_sharded(gt_annos, dt_annos, metric)
        # all_thresholds = -1.0 * dist_thresholds[metric, :, :, :] if metric == Metrics.BBOX_3D_NU_AP else \
        #     overlap_thresholds[metric, :, :, :]

        num_minoverlap = len(overlap_thresholds)
        num_classes = len(self.id_to_name)
        num_difficulties = 3

        precision = np.zeros([num_classes, num_difficulties, num_minoverlap, self.sample_points])
        recall = np.zeros([num_classes, num_difficulties, num_minoverlap, self.sample_points])
        instances_count = np.zeros([num_classes, num_difficulties])

        for class_idx in range(num_classes):
            for difficulty_idx in range(num_difficulties):
                gt_data_list, dt_data_list, ignored_gts, ignored_dets, dontcares, ignores_per_sample, \
                total_num_valid_gt = self.prepare_data(gt_annos, dt_annos, class_idx, difficulty_idx)
                instances_count[class_idx, difficulty_idx] = total_num_valid_gt

                for thresh_idx, min_overlap in enumerate(overlap_thresholds):
                    thresholds_list = []
                    for i in range(len(gt_annos)):
                        threshold = compute_threshold_jit(
                            overlaps[i],
                            gt_data_list[i],
                            dt_data_list[i],
                            ignored_gts[i],
                            ignored_dets[i],
                            min_overlap=min_overlap,
                            compute_fp=False
                        )
                        thresholds_list += threshold.tolist()
                    thresholds = np.array(
                        get_thresholds(np.array(thresholds_list), total_num_valid_gt, self.sample_points)
                    )
                    # TODO: Refactor hard coded numbers and strings
                    # [num_threshold, num_fields], fields: tp, fp, fn, aoe, aos, iou/dist error, -log(Probability,
                    # bev iou error)
                    pr = np.zeros([len(thresholds), 8])

                    idx = 0
                    for shard_idx, num_samples_per_shard in enumerate(shards):
                        gt_datas_part = np.concatenate(gt_data_list[idx:idx + num_samples_per_shard], 0)
                        dt_datas_part = np.concatenate(dt_data_list[idx:idx + num_samples_per_shard], 0)
                        dc_datas_part = np.concatenate(dontcares[idx:idx + num_samples_per_shard], 0)
                        ignored_dets_part = np.concatenate(ignored_dets[idx:idx + num_samples_per_shard], 0)
                        ignored_gts_part = np.concatenate(ignored_gts[idx:idx + num_samples_per_shard], 0)
                        fused_compute_statistics(
                            overlaps_by_shard[shard_idx],
                            pr,
                            total_gt_num[idx:idx + num_samples_per_shard],
                            total_dt_num[idx:idx + num_samples_per_shard],
                            ignores_per_sample[idx:idx + num_samples_per_shard],
                            gt_datas_part,
                            dt_datas_part,
                            dc_datas_part,
                            ignored_gts_part,
                            ignored_dets_part,
                            min_overlap=min_overlap,
                            thresholds=thresholds,
                            compute_angular_metrics=True
                        )
                        idx += num_samples_per_shard

                    for i in range(len(thresholds)):
                        recall[class_idx, difficulty_idx, thresh_idx, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                        precision[class_idx, difficulty_idx, thresh_idx, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])

        return {
            "recall": recall,
            "precision": precision,
        }

    def prepare_data(self, gt_annos, dt_annos, class_idx, difficulty_idx):
        """Wrapper function for cleaning data before computing metrics.
        """
        gt_list = []
        dt_list = []
        ignores_per_sample = []
        ignored_gts, ignored_dets, dontcares = [], [], []
        total_num_valid_gt = 0

        for gt_anno, dt_anno in zip(gt_annos, dt_annos):
            num_valid_gt, ignored_gt, ignored_det, ignored_bboxes = self.filter_data_fn(
                gt_anno, dt_anno, class_idx, difficulty_idx, self.id_to_name
            )
            ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
            ignored_dets.append(np.array(ignored_det, dtype=np.int64))

            if len(ignored_bboxes) == 0:
                ignored_bboxes = np.zeros((0, 4)).astype(np.float64)
            else:
                ignored_bboxes = np.stack(ignored_bboxes, 0).astype(np.float64)

            ignores_per_sample.append(ignored_bboxes.shape[0])
            dontcares.append(ignored_bboxes)
            total_num_valid_gt += num_valid_gt
            gt_list.append(
                np.concatenate([
                    gt_anno["bbox"], gt_anno["rotation_y"][..., np.newaxis], gt_anno["alpha"][..., np.newaxis],
                    gt_anno["dimensions"]
                ], 1)
            )

            dt_list.append(
                np.concatenate([
                    dt_anno["bbox"], dt_anno["rotation_y"][..., np.newaxis], dt_anno["alpha"][..., np.newaxis],
                    dt_anno["dimensions"], dt_anno["score"][..., np.newaxis]
                ], 1)
            )

        ignores_per_sample = np.stack(ignores_per_sample, axis=0)
        return gt_list, dt_list, ignored_gts, ignored_dets, dontcares, ignores_per_sample, total_num_valid_gt

    def calculate_match_degree_sharded(self, gt_annos, dt_annos, metric):
        assert len(gt_annos) == len(dt_annos)
        total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
        total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)

        overlaps_by_shard = []
        sample_idx = 0
        num_ground_truths = len(gt_annos)
        shards = self.get_shards(num_ground_truths, self.num_shards)

        for num_samples_per_shard in shards:
            gt_annos_part = gt_annos[sample_idx:sample_idx + num_samples_per_shard]
            dt_annos_part = dt_annos[sample_idx:sample_idx + num_samples_per_shard]

            if metric == 'BEV_AP':
                loc = np.concatenate([a["location"][:, [0, 2]] for a in gt_annos_part], 0)
                dims = np.concatenate([a["dimensions"][:, [0, 2]] for a in gt_annos_part], 0)
                rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
                gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
                loc = np.concatenate([a["location"][:, [0, 2]] for a in dt_annos_part], 0)
                dims = np.concatenate([a["dimensions"][:, [0, 2]] for a in dt_annos_part], 0)
                rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
                dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
                shard_match = self.bev_box_overlap(dt_boxes, gt_boxes).astype(np.float64)
            elif metric == "BOX3D_AP":
                loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
                dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
                rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
                gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
                loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
                dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
                rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
                dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
                shard_match = self.box_3d_overlap(dt_boxes, gt_boxes).astype(np.float64)
            else:
                raise ValueError("Unknown metric")

            # On each shard, we compute an IoU between all N predicted boxes and K GT boxes.
            # Shard overlap is a (N X K) array
            overlaps_by_shard.append(shard_match)

            sample_idx += num_samples_per_shard

        # Flatten into unsharded list
        overlaps = []
        sample_idx = 0
        for j, num_samples_per_shard in enumerate(shards):
            gt_num_idx, dt_num_idx = 0, 0
            for i in range(num_samples_per_shard):
                gt_box_num = total_gt_num[sample_idx + i]
                dt_box_num = total_dt_num[sample_idx + i]
                overlaps.append(
                    overlaps_by_shard[j][dt_num_idx:dt_num_idx + dt_box_num, gt_num_idx:gt_num_idx + gt_box_num, ]
                )
                gt_num_idx += gt_box_num
                dt_num_idx += dt_box_num
            sample_idx += num_samples_per_shard
        return overlaps, overlaps_by_shard, total_gt_num, total_dt_num

    def bev_box_overlap(self, boxes, qboxes, criterion=-1):
        """Compute overlap in BEV"""
        riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
        return riou

    def box_3d_overlap(self, boxes, qboxes, criterion=-1):
        """Compute 3D box IoU"""
        # For scale cuboid: use x, y to calculate bev iou, for kitti, use x, z to calculate bev iou
        rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]], qboxes[:, [0, 2, 3, 5, 6]], 2)
        d3_box_overlap_kernel(boxes, qboxes, rinc, criterion, True)
        return rinc


def clean_kitti_data(gt_anno, dt_anno, current_class, difficulty, id_to_name, difficulty_level_to_params=None):
    """Function for filtering KITTI data by difficulty and class.

    We filter with the following heuristics:
        If a ground truth matches the current class AND it falls below the difficulty
        threshold, we count it as a valid gt (append 0 in `ignored_gt` list).

        If a ground truth matches the current class but NOT the difficulty, OR it matches
        a class that is semantically too close to penalize (i.e. Van <-> Car),
        we ignore it (append 1 in `ignored_gt` list)

        If a ground truth doesn't belong to the current class, we ignore it (append -1 in `ignored_gt`)

        If a ground truth corresponds to a "DontCare" box, we append that box to the `ignored_bboxes` list.

        If a prediction matches the current class AND is above the minimum height threshold, we count it
        as a valid detection (append 0 in `ignored_dt`)

        If a prediction matches the current class AND it is too small, we ignore it (append 1 in `ignored_dt`)

        If a prediction doesn't belong to the class, we ignore it (append -1 in `ignored_dt`)

    Parameters
    ----------
    gt_anno: dict
        KITTI format ground truth. Please refer to note at the top for details on format.

    dt_anno: dict
        KITTI format prediction.  Please refer to note at the top for details on format.

    current_class: int
        Class ID, as int

    difficulty: int
        Difficulty: easy=0, moderate=1, difficult=2

    id_to_name: dict
        Mapping from class ID (int) to string name

    difficulty_level_to_params: dict default= None

    Returns
    -------
    num_valid_gt: int
        Number of valid ground truths

    ignored_gt: list[int]
        List of length num GTs. Populated as described above.

    ignored_dt: list[int]
        List of length num detections. Populated as described above.

    ignored_bboxes: list[np.ndarray]
        List of np.ndarray corresponding to boxes that are to be ignored
    """
    ignored_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = id_to_name[current_class].lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0

    for i in range(num_gt):
        bbox = gt_anno["bbox"][i]
        gt_name = gt_anno["name"][i].lower()
        height = bbox[3] - bbox[1]
        valid_class = -1

        # For KITTI, Van does not penalize car detections and person sitting does not penalize pedestrian
        if gt_name == current_cls_name:
            valid_class = 1
        elif current_cls_name == "Pedestrian".lower() and "Person_sitting".lower() == gt_name:
            valid_class = 0
        elif current_cls_name == "Car".lower() and "Van".lower() == gt_name:
            valid_class = 0
        else:
            valid_class = -1

        # Filter by occlusion/truncation
        ignore_for_truncation_occlusion = False
        if ((gt_anno["occluded"][i] > difficulty_level_to_params["max_occlusion"][difficulty])
            or (gt_anno["truncated"][i] > difficulty_level_to_params["max_truncation"][difficulty])
            or (height <= difficulty_level_to_params["min_height"][difficulty])):
            ignore_for_truncation_occlusion = True

        if valid_class == 1 and not ignore_for_truncation_occlusion:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif valid_class == 0 or (ignore_for_truncation_occlusion and (valid_class == 1)):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)

        # Track boxes are in "dontcare" areas
        if gt_name == "dontcare":
            ignored_bboxes.append(bbox)

    for i in range(num_dt):
        if dt_anno["name"][i].lower() == current_cls_name:
            valid_class = 1
        else:
            valid_class = -1
        height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])

        # If a box is too small, ignore it
        if height < difficulty_level_to_params["min_height"][difficulty]:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, ignored_bboxes


@numba.jit(nopython=True, fastmath=True)
def compute_threshold_jit(
    overlaps,
    gt_datas,
    dt_datas,
    ignored_gt,
    ignored_det,
    min_overlap,
    compute_fp=False,
):
    """Compute TP/FP statistics.
    Modified from https://github.com/sshaoehuai/PointRCNN/blob/master/tools/kitti_object_eval_python/eval.py
    """
    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]

    assigned_detection = [False] * det_size

    NO_DETECTION = np.finfo(np.float32).min
    tp, fp, fn = 0, 0, 0
    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0

    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]

            # Not hit during TP/FP computation
            if (not compute_fp and (overlap > min_overlap) and dt_score > valid_detection):
                assert not compute_fp, "For sanity, compute_fp shoudl be False if we are here"
                det_idx = j
                valid_detection = dt_score

        # No matched prediction found, valid GT
        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1

        # Matched prediction, but NO valid GT or matched prediction is too small so we ignore it (NOT BECAUSE THE
        # CLASS IS WRONG)
        elif ((valid_detection != NO_DETECTION) and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True

        # Matched prediction
        elif valid_detection != NO_DETECTION:
            tp += 1
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1

            assigned_detection[det_idx] = True

    return thresholds[:thresh_idx]


@numba.jit(nopython=True, fastmath=True)
def get_thresholds(scores, num_gt, num_sample_pts=41):
    """Get thresholds from a set of scores, up to num sample points

    Parameters
    ----------
    score: np.ndarray
        Numpy array of scores for predictions

    num_gt: int
        Number of ground truths

    num_sample_pts: int, default: 41
        Max number of thresholds on PR curve

    Returns
    -------
    threshold: np.ndarray
        Array of length 41, containing recall thresholds
    """
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall)) and (i < (len(scores) - 1))):
            continue
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


@numba.jit(nopython=True, fastmath=True)
def fused_compute_statistics(
    overlaps,
    pr,
    gt_nums,
    dt_nums,
    dc_nums,
    gt_datas,
    dt_datas,
    dontcares,
    ignored_gts,
    ignored_dets,
    min_overlap,
    thresholds,
    compute_angular_metrics=True,
):
    """Compute TP/FP statistics.
    Taken from https://github.com/sshaoehuai/PointRCNN/blob/master/tools/kitti_object_eval_python/eval.py
    without changes to avoid introducing errors"""

    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            # The key line that determines the ordering of the IoU matrix
            overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:gt_num + gt_nums[i]]

            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
            tp, fp, fn, error_yaw, similarity, _, match_degree, confidence_error, scale_error = \
                compute_statistics_jit(
                    overlap,
                    gt_data,
                    dt_data,
                    ignored_gt,
                    ignored_det,
                    dontcare,
                    min_overlap=min_overlap,
                    thresh=thresh,
                    compute_fp=True,
                    compute_angular_metrics=compute_angular_metrics)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            pr[t, 5] += match_degree
            pr[t, 6] += confidence_error
            pr[t, 7] += scale_error
            if error_yaw != -1:
                pr[t, 3] += error_yaw
            if similarity != -1:
                pr[t, 4] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


@numba.jit(nopython=True, fastmath=True)
def compute_statistics_jit(
    overlaps,
    gt_datas,
    dt_datas,
    ignored_gt,
    ignored_det,
    ignored_bboxes,
    min_overlap,
    thresh=0.0,
    compute_fp=False,
    compute_angular_metrics=False
):
    """Compute TP/FP statistics.
    Modified from https://github.com/sshaoehuai/PointRCNN/blob/master/tools/kitti_object_eval_python/eval.py
    """
    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    dt_yaws = dt_datas[:, 4]
    gt_yaws = gt_datas[:, 4]
    dt_alphas = dt_datas[:, 5]
    gt_alphas = gt_datas[:, 5]
    dt_bboxes = dt_datas[:, :4]
    gt_dimensions = gt_datas[:, 6:9]
    dt_dimensions = dt_datas[:, 6:9]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True

    NO_DETECTION = np.finfo(np.float32).min
    tp, fp, fn, error_yaw, similarity, match_degree, scale_error, confidence_error = 0, 0, 0, 0, 0, 0, 0, 0
    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0
    delta_yaw = np.zeros((gt_size, ))
    delta_alpha = np.zeros((gt_size, ))
    delta_idx = 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION

        max_overlap = np.finfo(np.float32).min
        target_scale_iou = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            overlap = overlaps[j, i]
            scale_iou = compute_scale_error(gt_dimensions[i, :], dt_dimensions[j, :])
            dt_score = dt_scores[j]

            # Not hit during TP/FP computation
            if (not compute_fp and (overlap > min_overlap) and dt_score > valid_detection):
                assert not compute_fp, "For sanity, compute_fp shoudl be False if we are here"
                det_idx = j
                valid_detection = dt_score
            elif (
                compute_fp and (overlap > min_overlap) and (overlap > max_overlap or assigned_ignored_det)
                and ignored_det[j] == 0
            ):
                max_overlap = overlap
                target_scale_iou = scale_iou
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False

            elif (compute_fp and (overlap > min_overlap) and (valid_detection == NO_DETECTION) and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        # No matched prediction found, valid GT
        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1

        # Matched prediction, but NO valid GT or matched prediction is too small so we ignore it (NOT BECAUSE THE
        # CLASS IS WRONG)
        elif ((valid_detection != NO_DETECTION) and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True

        # Matched prediction
        elif valid_detection != NO_DETECTION:
            tp += 1
            match_degree += abs(max_overlap)
            scale_error += 1.0 - abs(target_scale_iou)
            confidence_error += -math.log(dt_scores[det_idx])
            # Build a big list of all thresholds associated to true positives
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1

            if compute_angular_metrics:
                delta_yaw[delta_idx] = abs(angle_diff(float(gt_yaws[i]), float(dt_yaws[det_idx]), 2 * np.pi))
                delta_alpha[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx += 1

            assigned_detection[det_idx] = True

    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1 or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0
        fp -= nstuff
        if compute_angular_metrics:
            tmp_yaw = np.zeros((fp + delta_idx, ))
            tmp_alpha = np.zeros((fp + delta_idx, ))
            for i in range(delta_idx):
                tmp_yaw[i + fp] = delta_yaw[i]
                tmp_alpha[i + fp] = (1.0 + np.cos(delta_alpha[i])) / 2.0

            if tp > 0 or fp > 0:
                error_yaw = np.sum(tmp_yaw)
                similarity = np.sum(tmp_alpha)
            else:
                error_yaw = -1
                similarity = -1

    return tp, fp, fn, error_yaw, similarity, thresholds[:thresh_idx], match_degree, confidence_error, scale_error


@numba.jit(nopython=True)
def angle_diff(x, y, period):
    """Get the smallest angle difference between 2 angles: the angle from y to x.

    Parameters
    ----------
    x: float
        To angle.
    y: float
        From angle.
    period: float
        Periodicity in radians for assessing angle difference.

    Returns:
    ----------
    diff: float
        Signed smallest between-angle difference in range (-pi, pi).
    """

    # calculate angle difference, modulo to [0, 2*pi]
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff = diff - (2 * np.pi)  # shift (pi, 2*pi] to (-pi, 0]

    return diff


@numba.jit(nopython=True, fastmath=True)
def compute_scale_error(gt_dimension, dt_dimension):
    """
    This method compares predictions to the ground truth in terms of scale.
    It is equivalent to intersection over union (IOU) between the two boxes in 3D,
    if we assume that the boxes are aligned, i.e. translation and rotation are considered identical.

    Parameters
    ----------
    gt_dimension: List[float]
        GT annotation sample.
    dt_dimension: List[float]
        Predicted sample.

    Returns: float
    ----------
        Scale IOU.
    """

    # Compute IOU.
    min_wlh = [
        min(gt_dimension[0], dt_dimension[0]),
        min(gt_dimension[1], dt_dimension[1]),
        min(gt_dimension[2], dt_dimension[2])
    ]
    volume_gt = gt_dimension[0] * gt_dimension[1] * gt_dimension[2]
    volume_dt = dt_dimension[0] * dt_dimension[1] * dt_dimension[2]
    intersection = min_wlh[0] * min_wlh[1] * min_wlh[2]
    union = volume_gt + volume_dt - intersection
    iou = intersection / union

    return iou
