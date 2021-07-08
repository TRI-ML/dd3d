# Copyright 2021 Toyota Research Institute.  All rights reserved.
import itertools
import json
import logging
import os
import time
from collections import OrderedDict, defaultdict

from iopath.common.file_io import PathManager
from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes, load_gt, load_prediction
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.evaluate import DetectionEval as _DetectionEval
from pyquaternion import Quaternion

from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.structures.boxes import BoxMode
from detectron2.utils import comm as d2_comm

from tridet.data.datasets.nuscenes.build import ATTRIBUTE_IDS, CATEGORY_IDS, DATASET_NAME_TO_VERSION
from tridet.modeling.dd3d.postprocessing import get_group_idxs
from tridet.utils.comm import gather_dict, is_distributed

LOG = logging.getLogger(__name__)

BBOX3D_PREDICTION_FILE = "bbox3d_predictions.json"
NUSC_SUBMISSION_FILE = "nuscenes_submission.json"

NUM_IMAGES_PER_SAMPLE = 6
NUSCENES_DETECTION_CATEGORIES = list(CATEGORY_IDS.keys())

DEFAULT_ATTRIBUTES = {
    "car": "vehicle.moving",
    "bus": "vehicle.moving",
    "construction_vehicle": "vehicle.moving",
    "trailer": "vehicle.moving",
    "truck": "vehicle.moving",
    "bicycle": "cycle.with_rider",
    "motorcycle": "cycle.with_rider",
    "pedestrian": "pedestrian.moving"
}

DATASET_NAME_TO_EVAL_SET = {
    "nusc_train": "train",
    "nusc_val": "val",
    "nusc_val-subsample-8": "val",
    # "nusc_trainval": (build_d2_dicts_nuscenes, dict(split='trainval')),
    "nusc_test": "test",
    "nusc_mini_train": "mini_train",
    "nusc_mini_val": "mini_val",
    "nusc_train_detect": "train_detect",
    "nusc_train_track": "train_track"
}

VEH_ATTR_CLASSES = ("car", "bus", "construction_vehicle", "trailer", "truck")
PED_ATTR_CLASSES = ("pedestrian", )
CYC_ATTR_CLASSES = ("bicycle", "motorcycle")

VEH_ATTR_ID_TO_NAME = {0: 'vehicle.moving', 1: 'vehicle.parked', 2: 'vehicle.stopped'}
PED_ATTR_ID_TO_NAME = {0: 'pedestrian.moving', 1: 'pedestrian.standing', 2: 'pedestrian.sitting_lying_down'}
CYC_ATTR_ID_TO_NAME = {0: 'cycle.with_rider', 1: 'cycle.without_rider'}

for _id, name in VEH_ATTR_ID_TO_NAME.items():
    assert ATTRIBUTE_IDS[name] == _id
for _id, name in PED_ATTR_ID_TO_NAME.items():
    assert ATTRIBUTE_IDS[name] == _id
for _id, name in CYC_ATTR_ID_TO_NAME.items():
    assert ATTRIBUTE_IDS[name] == _id


class DetectionEval(_DetectionEval):
    """Patch DetectionEval of NuScenes devkit to only evaluate on samples in the predictions.
    """
    def __init__(self, nusc, config, result_path, eval_set, output_dir, verbose):
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(
            self.result_path, self.cfg.max_boxes_per_sample, DetectionBox, verbose=verbose
        )
        gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose)

        # assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
        #     "Samples in split doesn't match samples in predictions."

        # ----------------------------------------------------------------
        # Change from the original class:
        # Find and use the subset of GT boxes that matches with predictions.
        # ----------------------------------------------------------------
        assert set(self.pred_boxes.sample_tokens).issubset(set(gt_boxes.sample_tokens)), \
            "Samples in prediction must be a subset of samples in split."
        gt_boxes_subset = EvalBoxes()
        for token in self.pred_boxes.sample_tokens:
            gt_boxes_subset.add_boxes(token, gt_boxes[token])
        self.gt_boxes = gt_boxes_subset

        # Add center distances.
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = self.gt_boxes.sample_tokens


class NuscenesEvaluator(DatasetEvaluator):
    def __init__(self, nusc_root, dataset_name, output_dir):
        self._nusc_root = nusc_root
        self._dataset_name = dataset_name
        self._output_dir = output_dir
        self._only_make_submission_file = dataset_name == "nusc_test"

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        # List[Dict], each key'ed by category (str) + vectorized 3D box (10) + 2D box (4) + score (1) + file name (str)
        self._predictions_as_json = []
        self._nusc_sample_results = defaultdict(list)

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
        sample_tokens = [x['sample_token'] for x in inputs]
        idx_to_token = get_group_idxs(sample_tokens, NUM_IMAGES_PER_SAMPLE, inverse=True)

        # This handles samples with no detections.
        for token in set(sample_tokens):
            self._nusc_sample_results[token]  # pylint: disable=pointless-statement

        for image_idx, (input_per_image, pred_per_image) in enumerate(zip(inputs, outputs)):
            pred_classes = pred_per_image['instances'].pred_classes
            pred_boxes = pred_per_image['instances'].pred_boxes.tensor
            pred_boxes3d = pred_per_image['instances'].pred_boxes3d
            # pred_boxes3d = pred_per_image['instances'].pred_box3d_as_vec
            scores = pred_per_image['instances'].scores
            scores_3d = pred_per_image['instances'].scores_3d

            pred_attributes = pred_per_image['instances'].pred_attributes
            pred_speeds = pred_per_image['instances'].pred_speeds

            file_name = input_per_image['file_name']
            image_id = input_per_image['image_id']

            pred_boxes3d_global = pred_per_image['instances'].pred_boxes3d_global

            sample_token = idx_to_token[image_idx]

            for class_id, box3d, box3d_global, score_3d, box2d, attr, speed, score in zip(
                pred_classes, pred_boxes3d, pred_boxes3d_global, scores_3d, pred_boxes, pred_attributes, pred_speeds,
                scores
            ):
                # class_name = self._metadata.thing_classes[class_id]
                class_name = NUSCENES_DETECTION_CATEGORIES[class_id]

                # attribute
                attr = attr.item()
                if class_name in VEH_ATTR_CLASSES:
                    attr_name = VEH_ATTR_ID_TO_NAME[attr % len(VEH_ATTR_ID_TO_NAME)]
                elif class_name in PED_ATTR_CLASSES:
                    attr_name = PED_ATTR_ID_TO_NAME[attr % len(PED_ATTR_ID_TO_NAME)]
                elif class_name in CYC_ATTR_CLASSES:
                    attr_name = CYC_ATTR_ID_TO_NAME[attr % len(CYC_ATTR_ID_TO_NAME)]
                else:
                    attr_name = ""

                # velocity
                vel_global = speed.cpu().numpy() * Quaternion(box3d_global.quat.tolist()[0]).rotation_matrix.T[0]
                vx, vy = vel_global[:2].tolist()
                # speed * Quaternion(ann['rotation']).rotation_matrix.T[0] ~= vel_global

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

                nusc_det = self.build_nusc_detection(
                    sample_token, box3d_global, class_name, score_3d, attribute=attr_name, velocity=[vx, vy]
                )
                self._nusc_sample_results[sample_token].append(nusc_det)

    @staticmethod
    def build_nusc_detection(sample_token, box3d_global, category, score, attribute=None, velocity=None):
        box3d_vec = box3d_global.vectorize().tolist()[0]
        if attribute is None:
            attribute = DEFAULT_ATTRIBUTES.get(category, '')
        if velocity is None:
            velocity = [0., .0]
        res = {
            "sample_token": sample_token,
            "rotation": box3d_vec[:4],
            "translation": box3d_vec[4:7],
            "size": box3d_vec[7:],
            "detection_name": category,
            "detection_score": score.item(),
            "attribute_name": attribute,
            "velocity": velocity
        }
        return res

    def evaluate(self):
        if is_distributed():
            d2_comm.synchronize()

            # NOTE: The gather cmd seems to not work well for large objects. For now, use non-distributed eval, if program hangs.
            LOG.info("Gathering predictions_as_json...")
            predictions_as_json = d2_comm.gather(self._predictions_as_json, dst=0)
            LOG.info("done..")
            predictions_as_json = list(itertools.chain(*predictions_as_json))

            LOG.info("Gathering nusc_sample_results...")
            nusc_sample_results = gather_dict(self._nusc_sample_results)
            LOG.info("done..")

            if not d2_comm.is_main_process():
                return

        else:
            predictions_as_json = self._predictions_as_json
            nusc_sample_results = self._nusc_sample_results

        eval_meta = {
            'use_camera': True,
            'use_lidar': False,
            'use_radar': False,
            'use_map': False,
            'use_external': True,
        }

        PathManager().mkdirs(self._output_dir)
        file_path = os.path.join(self._output_dir, BBOX3D_PREDICTION_FILE)
        with open(file_path, 'w') as f:
            json.dump(predictions_as_json, f, indent=4)

        nusc_submission_file_path = os.path.join(self._output_dir, NUSC_SUBMISSION_FILE)
        nusc_submission = {'meta': eval_meta, 'results': nusc_sample_results}
        with open(nusc_submission_file_path, 'w') as f:
            json.dump(nusc_submission, f, indent=4)

        if self._only_make_submission_file:
            return {}

        nusc_eval_cfg = config_factory('detection_cvpr_2019')
        nusc = NuScenes(version=DATASET_NAME_TO_VERSION[self._dataset_name], dataroot=self._nusc_root, verbose=True)

        nusc_eval = DetectionEval(
            nusc,
            nusc_eval_cfg,
            nusc_submission_file_path,
            eval_set=DATASET_NAME_TO_EVAL_SET[self._dataset_name],
            output_dir=self._output_dir,
            verbose=False
        )
        LOG.info("Running NuScenes evaluation...")
        st = time.time()
        metrics, _ = nusc_eval.evaluate()
        took = time.time() - st
        LOG.info(f"Done (took {took:.2f}s).")

        nusc_metrics = metrics.serialize()
        nusc_metrics.pop('cfg')
        nusc_metrics.pop('eval_time')

        return nusc_metrics
