#!/usr/bin/env python
# Copyright 2021 Toyota Research Institute.  All rights reserved.
import logging
import sys
from collections import defaultdict

import cv2
import hydra
from tqdm import tqdm

from detectron2.data import MetadataCatalog

from tridet.data import build_test_dataloader, build_train_dataloader
from tridet.data.dataset_mappers import get_dataset_mapper
from tridet.data.datasets import register_datasets
from tridet.utils.setup import setup
from tridet.utils.visualization import mosaic
from tridet.visualizers import get_dataloader_visualizer

LOG = logging.getLogger('tridet')


@hydra.main(config_path="../configs/", config_name="visualize_dataloader")
def main(cfg):
    setup(cfg)
    dataset_names = register_datasets(cfg)
    if cfg.ONLY_REGISTER_DATASETS:
        return {}, cfg
    LOG.info(f"Registered {len(dataset_names)} datasets:" + '\n\t' + '\n\t'.join(dataset_names))

    if cfg.USE_TEST:
        dataset_name = cfg.DATASETS.TEST.NAME
        mapper = get_dataset_mapper(cfg, is_train=False)
        dataloader, _ = build_test_dataloader(cfg, dataset_name, mapper=mapper)
    else:
        mapper = get_dataset_mapper(cfg, is_train=True)
        dataloader, _ = build_train_dataloader(cfg, mapper=mapper)

    visualizer_names = MetadataCatalog.get(cfg.DATASETS.TRAIN.NAME).loader_visualizers
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        viz_images = defaultdict(dict)
        LOG.info("Press any key to continue, press 'q' to quit.")
        for viz_name in visualizer_names:
            viz = get_dataloader_visualizer(cfg, viz_name, cfg.DATASETS.TRAIN.NAME)
            for idx, x in enumerate(batch):
                viz_images[idx].update(viz.visualize(x))

        for k in range(len(batch)):
            gt_viz = mosaic(list(viz_images[k].values()))
            cv2.imshow("dataloader", gt_viz[:, :, ::-1])

            if cv2.waitKey(0) & 0xFF == ord('q'):
                sys.exit()


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
