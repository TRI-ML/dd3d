# Copyright 2021 Toyota Research Institute.  All rights reserved.
import logging
import os
from functools import partial

from detectron2.data import DatasetCatalog
from detectron2.utils.comm import get_world_size

from tridet.data.datasets.nuscenes.build import build_nuscenes_dataset, register_nuscenes_metadata

LOG = logging.getLogger(__name__)

NUSCENES_ROOT = "nuScenes"

NUSC_DATASET_NAMES = [
    "nusc_train",
    "nusc_val",
    "nusc_val-subsample-8",
    "nusc_trainval",
    "nusc_test",
    "nusc_mini_train",
    "nusc_mini_val",
]

DATASET_DICTS_BUILDER = {name: (build_nuscenes_dataset, dict(name=name)) for name in NUSC_DATASET_NAMES}

METADATA_BUILDER = {name: (register_nuscenes_metadata, {}) for name in DATASET_DICTS_BUILDER.keys()}


def register_nuscenes_datasets(required_datasets, cfg):
    if cfg.DATASETS.TEST.NAME in ("nusc_train", "nusc_val", "nusc_trainval", "nusc_test") and \
        get_world_size() > 1:
        raise LOG.warning("The distributed evaluation does not work well with large test set for now. " \
            f"If program hangs, consider using non-distributed evaluation: {cfg.DATASETS.TEST.NAME}")

    nusc_datasets = sorted(list(set(required_datasets).intersection(DATASET_DICTS_BUILDER.keys())))
    if nusc_datasets:
        LOG.info(f"nuScenes-3D dataset(s): {', '.join(nusc_datasets)} ")
        for name in nusc_datasets:
            fn, kwargs = DATASET_DICTS_BUILDER[name]
            kwargs.update({
                'root_dir': os.path.join(cfg.DATASET_ROOT, NUSCENES_ROOT),
                'min_num_lidar_points': cfg.DATASETS.TRAIN.MIN_NUM_LIDAR_PTS,
                'min_box_visibility': cfg.DATASETS.TRAIN.MIN_BOX_VISIBILITY
            })
            DatasetCatalog.register(name, partial(fn, **kwargs))

            fn, kwargs = METADATA_BUILDER[name]
            fn(name, **kwargs)
    return nusc_datasets
