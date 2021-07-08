# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2021 Toyota Research Institute.  All rights reserved.
# Adapted from detectron2:
#   https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/coco.py
import datetime
import json
import logging
import os
import shutil

import numpy as np
import pycocotools.mask as mask_util

from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager

from tridet.utils.comm import broadcast_from_master

LOG = logging.getLogger(__name__)


def convert_to_coco_dict(dataset_dicts, metadata):
    """
    Convert an instance detection/segmentation or keypoint detection dataset
    in detectron2's standard format into COCO json format.

    Generic dataset description can be found here:
    https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset

    COCO data format description can be found here:
    http://cocodataset.org/#format-data

    Args:
        dataset_name (str):
            name of the source dataset
            Must be registered in DatastCatalog and in detectron2's standard format.
            Must have corresponding metadata "thing_classes"
    Returns:
        coco_dict: serializable dict in COCO json format
    """
    # unmap the category mapping ids for COCO
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        reverse_id_mapping = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}
        reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]  # noqa
    else:
        reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa

    categories = [{"id": reverse_id_mapper(id), "name": name} for id, name in enumerate(metadata.thing_classes)]

    LOG.info("Converting dataset dicts into COCO format")
    coco_images = []
    coco_annotations = []

    for image_id, image_dict in enumerate(dataset_dicts):
        coco_image = {
            "id": image_dict.get("image_id", image_id),
            "width": int(image_dict["width"]),
            "height": int(image_dict["height"]),
            "file_name": str(image_dict["file_name"]),
        }
        coco_images.append(coco_image)

        anns_per_image = image_dict.get("annotations", [])
        for annotation in anns_per_image:
            # create a new dict with only COCO fields
            coco_annotation = {}

            # COCO requirement: XYWH box format for axis-align and XYWHA for rotated
            bbox = annotation["bbox"]
            if isinstance(bbox, np.ndarray):
                if bbox.ndim != 1:
                    raise ValueError(f"bbox has to be 1-dimensional. Got shape={bbox.shape}.")
                bbox = bbox.tolist()
            if len(bbox) not in [4, 5]:
                raise ValueError(f"bbox has to has length 4 or 5. Got {bbox}.")
            from_bbox_mode = annotation["bbox_mode"]
            to_bbox_mode = BoxMode.XYWH_ABS if len(bbox) == 4 else BoxMode.XYWHA_ABS
            bbox = BoxMode.convert(bbox, from_bbox_mode, to_bbox_mode)

            # COCO requirement: instance area
            if "segmentation" in annotation:
                # Computing areas for instances by counting the pixels
                segmentation = annotation["segmentation"]
                # TODO: check segmentation type: RLE, BinaryMask or Polygon
                if isinstance(segmentation, list):
                    polygons = PolygonMasks([segmentation])
                    area = polygons.area()[0].item()
                elif isinstance(segmentation, dict):  # RLE
                    area = mask_util.area(segmentation).item()
                else:
                    raise TypeError(f"Unknown segmentation type {type(segmentation)}!")
            else:
                # Computing areas using bounding boxes
                if to_bbox_mode == BoxMode.XYWH_ABS:
                    bbox_xy = BoxMode.convert(bbox, to_bbox_mode, BoxMode.XYXY_ABS)
                    area = Boxes([bbox_xy]).area()[0].item()
                else:
                    area = RotatedBoxes([bbox]).area()[0].item()

            if "keypoints" in annotation:
                keypoints = annotation["keypoints"]  # list[int]
                for idx, v in enumerate(keypoints):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # For COCO format consistency we substract 0.5
                        # https://github.com/facebookresearch/detectron2/pull/175#issuecomment-551202163
                        keypoints[idx] = v - 0.5
                if "num_keypoints" in annotation:
                    num_keypoints = annotation["num_keypoints"]
                else:
                    num_keypoints = sum(kp > 0 for kp in keypoints[2::3])

            # COCO requirement:
            #   linking annotations to images
            #   "id" field must start with 1
            coco_annotation["id"] = len(coco_annotations) + 1
            coco_annotation["image_id"] = coco_image["id"]
            coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]
            coco_annotation["area"] = float(area)
            coco_annotation["iscrowd"] = int(annotation.get("iscrowd", 0))
            coco_annotation["category_id"] = int(reverse_id_mapper(annotation["category_id"]))

            # Add optional fields
            if "keypoints" in annotation:
                coco_annotation["keypoints"] = keypoints
                coco_annotation["num_keypoints"] = num_keypoints

            if "segmentation" in annotation:
                seg = coco_annotation["segmentation"] = annotation["segmentation"]
                if isinstance(seg, dict):  # RLE
                    counts = seg["counts"]
                    if not isinstance(counts, str):
                        # make it json-serializable
                        seg["counts"] = counts.decode("ascii")

            coco_annotations.append(coco_annotation)

    LOG.info("Conversion finished, " f"#images: {len(coco_images)}, #annotations: {len(coco_annotations)}")

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file for Detectron2.",
    }
    coco_dict = {"info": info, "images": coco_images, "categories": categories, "licenses": None}
    if len(coco_annotations) > 0:
        coco_dict["annotations"] = coco_annotations
    return coco_dict


def convert_to_coco_json(dataset_dicts, metadata, dataset_name, output_file, allow_cached=True):
    """
    Copy of https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/coco.py#L445,
    but removing the `file_lock`.

    ---------------------------------------------------------------------------------------------------------

    Converts dataset into COCO format and saves it to a json file.
    dataset_name must be registered in DatasetCatalog and in detectron2's standard format.

    Args:
        dataset_name:
            reference from the config file to the catalogs
            must be registered in DatasetCatalog and in detectron2's standard format
        output_file: path of json file that will be saved to
        allow_cached: if json file is already present then skip conversion
    """

    # TODO: The dataset or the conversion script *may* change,
    # a checksum would be useful for validating the cached data

    PathManager.mkdirs(os.path.dirname(output_file))
    if PathManager.exists(output_file) and allow_cached:
        LOG.warning(
            f"Using previously cached COCO format annotations at '{output_file}'. "
            "You need to clear the cache file if your dataset has been modified."
        )
    else:
        LOG.info(f"Converting annotations of dataset '{dataset_name}' to COCO format ...)")
        coco_dict = convert_to_coco_dict(dataset_dicts, metadata)

        LOG.info(f"Caching COCO format annotations at '{output_file}' ...")
        tmp_file = output_file + ".tmp"
        with PathManager.open(tmp_file, "w") as f:
            json.dump(coco_dict, f)
        shutil.move(tmp_file, output_file)


@broadcast_from_master
def create_coco_format_cache(dataset_dicts, metadata, dataset_name, cache_dir):
    cache_path = os.path.join(cache_dir, f"{dataset_name}_coco_format.json")
    convert_to_coco_json(dataset_dicts, metadata, dataset_name, cache_path)
    return cache_path
