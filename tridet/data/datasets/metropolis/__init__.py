# Copyright 2021 Toyota Research Institute.  All rights reserved.
import logging
import os
from functools import partial

from detectron2.data import DatasetCatalog

from tridet.data.datasets.metropolis.build import build_metropolis_dataset, register_metropolis_metadata

LOG = logging.getLogger(__name__)

METROPOLIS_ROOT = "Metropolis"  # TODO:

METROPOLIS_DATASET_NAMES = [
    "train",
    "val",
    "test",
]

DATASET_DICTS_BUILDER = {name: (build_metropolis_dataset, dict(name=name)) for name in METROPOLIS_DATASET_NAMES}

METADATA_BUILDER = {name: (register_metropolis_metadata, {}) for name in DATASET_DICTS_BUILDER.keys()}


def register_nuscenes_datasets(required_datasets, cfg):

    metropolis_datasets = sorted(list(set(required_datasets).intersection(DATASET_DICTS_BUILDER.keys())))
    if metropolis_datasets:
        LOG.info(f"Metropolis dataset(s): {', '.join(metropolis_datasets)} ")
        for name in metropolis_datasets:
            fn, kwargs = DATASET_DICTS_BUILDER[name]
            kwargs.update({
                'root_dir': os.path.join(cfg.DATASET_ROOT, METROPOLIS_ROOT),
                'min_num_lidar_points': cfg.DATASETS.TRAIN.MIN_NUM_LIDAR_PTS,
                'min_box_visibility': cfg.DATASETS.TRAIN.MIN_BOX_VISIBILITY
            })
            DatasetCatalog.register(name, partial(fn, **kwargs))

            fn, kwargs = METADATA_BUILDER[name]
            fn(name, **kwargs)
    return metropolis_datasets
