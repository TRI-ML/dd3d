"""
Metropolis dataset builder
"""
import functools
from collections import OrderedDict
from typing import Tuple, List, Optional, Dict, Any, Union

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from detectron2.data import MetadataCatalog
from detectron2.structures.boxes import BoxMode
from metropolis import Metropolis
from metropolis.utils.data_classes import Box, Box2d, EquiBox2d
from metropolis.utils.color_map import get_colormap

from tridet.data import collect_dataset_dicts
from tridet.structures.boxes3d import GenericBoxes3D
from tridet.structures.pose import Pose
from tridet.utils.geometry import project_points3d


ATTRIBUTE_IDS = {  # TODO: set attributes ids
    'ambiguous': 0,
    'out-of-frame': 0,
    'occluded.not': 5,
    'occluded.partially': 6,
    'occluded.fully': 7,
    'front-back.front': 8,
    'front-back.back': 9,
    'front-back.ambiguous': 0,
    'vehicle-state.other': 4,
    'vehicle-state.moving': 3,
    'vehicle-state.parked': 2,
    'vehicle-state.stopped': 1,
    'vehicle-state.ambiguous': 0,
}

# CAMERA_NAMES = ('CAM_EQUIRECTANGULAR', )
CAMERA_NAMES = ('CAM_FRONT', 'CAM_LEFT', 'CAM_RIGHT', 'CAM_BACK')  # TODO: Process all cameras

MAX_NUM_ATTRIBUTES = 10

CATEGORY_IDS = OrderedDict({    # TODO: choose appropriate categories
    'human--person': 0,
    'human--person--group': 1,
    'human--rider--bicyclist': 2,
    'object--vehicle--bicycle': 3,
    'object--vehicle--bus': 4,
    'object--vehicle--car': 5,
    'object--vehicle--motorcycle': 6,
    'object--vehicle--truck': 7,
    'object--vehicle--group': 8,
    'object--vehicle--other-vehicle': 9
})


def _build_id(scene_name, sample_idx, datum_name):
    sample_id = f"{scene_name}_{sample_idx:03d}"
    image_id = f"{sample_id}_{datum_name}"
    return image_id, sample_id


class MetropolisDataset(Dataset):
    def __init__(self, name: str, root_dir: str, datum_names: List[str] = CAMERA_NAMES):
        self.name = name
        self.met = Metropolis(name, root_dir)
        self.datum_names = datum_names

        self.dataset_item_info = self._build_dataset_item_info()

    def __len__(self):
        return len(self.dataset_item_info)

    def get_instance_annotations(self,
                box_3d_list: List[Box],
                box_2d_list: Union[List[Box2d], List[EquiBox2d]],
                cam_intrinsic: np.array,
                image_shape: Tuple[int, int]) -> List[OrderedDict]:

        annotations = []
        for box_3d, box_2d in zip(box_3d_list, box_2d_list):
            # assert box_3d.token == box_2d.token # TODO: doesnt match. Fix
            sample_annotation = self.met.get('sample_annotation_2d', box_2d.token)  # TODO: can be called before for loop?
            # sample_annotation_2d = self.met.get('sample_annotation_2d', box_2d.token)

            annotation = OrderedDict()

            # --------
            # Category
            # --------
            # category_id = CATEGORY_IDS.get(box_3d.name)
            # if category_id is None:
            #     continue
            # NOTE: We can define new index for OTHER, if category is not specified. Ex:
            category_id = CATEGORY_IDS.get(box_3d.name, max(CATEGORY_IDS.values()))
            annotation['category_id'] = category_id
            R = box_3d.rotation_matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = box_3d.center
            T_inv = np.linalg.inv(T)
            tvec = T_inv[:3, 3]
            # ------
            # 3D box
            # ------
            bbox3d = GenericBoxes3D(box_3d.orientation, box_3d.center, box_3d.lwh[[1, 0, 2]] )  # lwh -> wlh # TODO: Check 
            annotation['bbox3d'] = bbox3d.vectorize().tolist()[0]
            # annotation['bbox3d'] = None

            # -------
            # 2D box
            # -------
            corners = project_points3d(bbox3d.corners.cpu().numpy().squeeze(0), cam_intrinsic)
            l, t = corners[:, 0].min(), corners[:, 1].min()
            r, b = corners[:, 0].max(), corners[:, 1].max()

            x1 = max(0, l)
            y1 = max(0, t)
            x2 = min(image_shape[1], r)
            y2 = min(image_shape[0], b)

            annotation['bbox'] = [x1, y1, x2, y2]  # TODO: Fix points for EquiBox2d
            annotation['bbox_mode'] = BoxMode.XYXY_ABS

            # --------
            # Track ID
            # --------
            annotation['track_id'] = self.met.getind('instance', sample_annotation['instance_token'])
            #if any(annotation['track_id'] == ann['track_id'] for ann in annotations):
            #    continue
            # ---------
            # Attribute
            # ---------
            # attr_tokens = sample_annotation_2d['attribute_tokens']  # TODO: Fix attribute use. There are more than 1
            # attribute_id = MAX_NUM_ATTRIBUTES
            # if attr_tokens:
            #     attribute = self.met.get('attribute', attr_tokens[0])['name']
            #     attribute_id = ATTRIBUTE_IDS[attribute]
            # annotation['attribute_id'] = attribute_id

            annotations.append(annotation)

        return annotations

    def __getitem__(self, idx: int) -> OrderedDict:

        datum_token, sample_token, scene_name, sample_idx, datum_name = self.dataset_item_info[idx]
        datum = self.met.get('sample_data', datum_token)

        filename, box_3d_list, box_2d_list, K = self.met.get_sample_data(datum_token)
        image_id, sample_id = _build_id(scene_name, sample_idx, datum_name)
        height, width = datum['height'], datum['width']

        d2_dict = OrderedDict(
            file_name=filename,
            height=height,
            width=width,
            image_id=image_id,
            sample_id=sample_id,
            sample_token=sample_token
        )

        # Intrinsics
        d2_dict['intrinsics'] = list(K.flatten())

        # Get pose of the sensor (S) from vehicle (V) frame
        _pose_VS = self.met.get('calibrated_sensor', datum['calibrated_sensor_token'])
        pose_VS = Pose(wxyz=np.float64(_pose_VS['rotation']), tvec=np.float64(_pose_VS['translation']))

        # Get ego-pose of the vehicle (V) from global/world (W) frame
        _pose_WV = self.met.get('ego_pose', datum['ego_pose_token'])
        pose_WV = Pose(wxyz=np.float64(_pose_WV['rotation']), tvec=np.float64(_pose_WV['translation']))
        pose_WS = pose_WV * pose_VS

        d2_dict['pose'] = {'wxyz': list(pose_WS.quat.elements), 'tvec': list(pose_WS.tvec)}
        d2_dict['extrinsics'] = {'wxyz': list(pose_VS.quat.elements), 'tvec': list(pose_VS.tvec)}

        # Get annotations for each box
        d2_dict['annotations'] = self.get_instance_annotations(box_3d_list, box_2d_list, K, (height, width))

        return d2_dict

    def _build_dataset_item_info(self) -> List[Tuple[str, str, str, int, str]]:
        dataset_items = []

        for scene in tqdm(self.met.scene):
            sample_token = scene['first_sample_token']
            for sample_idx in range(scene['nbr_samples']):
                sample = self.met.get('sample', sample_token)
                for datum_name, datum_token in sample['data'].items():
                    if datum_name not in self.datum_names:
                        continue
                    dataset_items.append((datum_token, sample_token, scene['name'], sample_idx, datum_name))
                sample_token = sample['next_sample']

        return dataset_items

    def __print_categories_with_instance(self):  # NOTE: Debug
        has_instance = 0
        for cat in self.met.category:
            if cat['has_instances']:
                print(f"{cat['name']}:  {cat['description']}")
                has_instance += 1
        print(has_instance)

    def __print_attributes(self):  # NOTE: Debug
        for attr in self.met.attribute:
            print(f"{attr['name']}:  {attr['description']}")


@functools.lru_cache(maxsize=1000)
def build_metropolis_dataset(name, root_dir):
    dataset = MetropolisDataset(name, root_dir)
    dataset_dicts = collect_dataset_dicts(dataset)
    return dataset_dicts


def register_metropolis_metadata(dataset_name):
    metadata = MetadataCatalog.get(dataset_name)
    metadata.thing_classes = list(CATEGORY_IDS.keys())

    colormap = get_colormap()
    metadata.thing_colors = [colormap[klass] for klass in metadata.thing_classes]

    metadata.id_to_name = {idx: klass for idx, klass in enumerate(metadata.thing_classes)}
    metadata.contiguous_id_to_name = {idx: klass for idx, klass in enumerate(metadata.thing_classes)}
    metadata.name_to_contiguous_id = {name: idx for idx, name in metadata.contiguous_id_to_name.items()}

    # metadata.evaluators = ("metropolis_evaluator", ) # TODO: implement evaluator
    metadata.pred_visualizers = ("box3d_visualizer", )
    metadata.loader_visualizers = ("box3d_visualizer", )
