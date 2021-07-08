# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2021 Toyota Research Institute.  All rights reserved.
# Adapted from detectron2:
#   https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/events.py
import wandb
from detectron2.utils.events import EventStorage

from tridet.utils.comm import master_only


class WandbEventStorage(EventStorage):

    @master_only
    def put_scalar(self, name, value, smoothing_hint=True, wandb_log=True):
        super().put_scalar(name, value, smoothing_hint=smoothing_hint)

        # Add W&B logging
        name = self._current_prefix + name
        value = float(value)
        if wandb_log and wandb.run:
            wandb.log({name: value}, step=self.iter)
