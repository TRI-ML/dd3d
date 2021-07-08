# Copyright 2021 Toyota Research Institute.  All rights reserved.
import tridet.modeling.dd3d
from tridet.modeling import feature_extractor
from tridet.modeling.dd3d import DD3DWithTTA, NuscenesDD3DWithTTA

TTA_MODELS = {
    "DD3D": DD3DWithTTA,
    "NuscenesDD3D": NuscenesDD3DWithTTA,
}


def build_tta_model(cfg, model):
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    assert meta_arch in TTA_MODELS, f"Test-time augmentation model is not available: {meta_arch}"
    return TTA_MODELS[meta_arch](cfg, model)
