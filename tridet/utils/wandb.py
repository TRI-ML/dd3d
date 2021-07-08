# Copyright 2021 Toyota Research Institute.  All rights reserved.
import logging
import os
from collections import OrderedDict
from collections.abc import Mapping
from functools import wraps

import wandb
from detectron2.utils.events import get_event_storage
from omegaconf import OmegaConf

from tridet.utils.comm import broadcast_from_master, master_only

LOG = logging.getLogger(__name__)


def wandb_credential_is_available():
    if os.environ.get('WANDB_API_KEY', None):
        return True
    else:
        return False


@master_only
def init_wandb(cfg):
    if not wandb_credential_is_available():
        LOG.warning(
            "W&B credential must be defined in environment variables."
            "Use `WANDB.ENABLED=False` to suppress this warning. "
            "Skipping `init_wandb`..."
        )
        return

    if cfg.WANDB.DRYRUN:
        os.environ['WANDB_MODE'] = 'dryrun'

    _cfg = cfg.copy()
    del _cfg.hydra
    cfg_as_dict = OmegaConf.to_container(_cfg, resolve=True)
    wandb.init(project=cfg.WANDB.PROJECT, config=cfg_as_dict, tags=cfg.WANDB.TAGS, group=cfg.WANDB.GROUP)


def wandb_is_initialized():
    try:
        wandb.run.id  # pylint: disable=pointless-statement
        initialized = True
    except AttributeError:
        initialized = False
    return initialized


def if_wandb_initialized(fn):
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if wandb_is_initialized():
            return fn(*args, **kwargs)
        else:
            return None

    return wrapped_fn


@broadcast_from_master
def derive_output_dir_from_wandb_id(cfg):
    assert wandb_is_initialized()
    wandb_run_dir = wandb.run.dir
    if wandb_run_dir.endswith('/files'):  # wandb 0.10.x
        wandb_run_dir = wandb_run_dir[:-6]
    datetime_str, wandb_run_id = wandb_run_dir.split('-')[-2:]
    assert wandb_run_id == wandb.run.id

    output_dir = os.path.join(cfg.OUTPUT_ROOT, '-'.join([wandb_run_id, datetime_str]))
    return output_dir


@master_only
@if_wandb_initialized
def log_nested_dict(dikt):
    storage = get_event_storage()
    step = storage.iter

    wandb.log(flatten_dict(dikt), step=step)


def flatten_dict(results):
    """
    Almost identical to detectron2.evaluation.testing:flatten_result_dict()', but using 'OrderedDict'
    --------------------------------------------------------------------------------------------------

    Expand a hierarchical dict of scalars into a flat dict of scalars.
    If results[k1][k2][k3] = v, the returned dict will have the entry
    {"k1/k2/k3": v}.

    Args:
        results (dict):
    """
    r = OrderedDict()
    for k, v in results.items():
        k = str(k)
        if isinstance(v, Mapping):
            v = flatten_dict(v)
            for kk, vv in v.items():
                r[k + "/" + kk] = vv
        else:
            r[k] = v
    return r
