# Copyright 2021 Toyota Research Institute.  All rights reserved.
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog

from tridet.visualizers.box3d_visualizer import Box3DDataloaderVisualizer, Box3DPredictionVisualizer
from tridet.visualizers.d2_visualizer import D2DataloaderVisualizer, D2PredictionVisualizer


def get_predictions_visualizer(cfg, visualizer_name, dataset_name, inference_output_dir):
    if visualizer_name == 'd2_visualizer':
        return D2PredictionVisualizer(cfg, dataset_name, inference_output_dir)
    elif visualizer_name == "box3d_visualizer":
        return Box3DPredictionVisualizer(cfg, dataset_name, inference_output_dir)
    else:
        raise ValueError(f"Invalid visualizer: {visualizer_name}")


def get_dataloader_visualizer(cfg, visualizer_name, dataset_name):
    if visualizer_name == 'd2_visualizer':
        return D2DataloaderVisualizer(cfg, dataset_name)
    elif visualizer_name == "box3d_visualizer":
        return Box3DDataloaderVisualizer(cfg, dataset_name)
    else:
        raise ValueError(f"Invalid visualizer: {visualizer_name}")
