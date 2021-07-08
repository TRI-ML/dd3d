# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2021 Toyota Research Institute.  All rights reserved.
import json
import logging
import os
import resource
from datetime import datetime

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

import detectron2.utils.comm as d2_comm
from detectron2.utils.env import seed_all_rng
from detectron2.utils.events import _CURRENT_STORAGE_STACK

from tridet.utils.comm import broadcast_from_master
from tridet.utils.events import WandbEventStorage

LOG = logging.getLogger(__name__)


def setup_distributed(world_size, rank):
    """
    Adapted from detectron2:
        https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/launch.py#L85
    """
    host = os.environ["MASTER_ADDR"] if "MASTER_ADDR" in os.environ else "127.0.0.1"
    port = 12345
    dist_url = f"tcp://{host}:{port}"
    try:
        dist.init_process_group(backend='NCCL', init_method=dist_url, world_size=world_size, rank=rank)
    except Exception as e:
        logging.error("Process group URL: %s", dist_url)
        raise e
    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    d2_comm.synchronize()

    # Assumption: all machines have the same number of GPUs.
    num_gpus_per_machine = torch.cuda.device_count()
    machine_rank = rank // num_gpus_per_machine

    # Setup the local process group (which contains ranks within the same machine)
    assert d2_comm._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            d2_comm._LOCAL_PROCESS_GROUP = pg

    # Declare GPU device.
    local_rank = rank % num_gpus_per_machine
    torch.cuda.set_device(local_rank)

    # Multi-node training often fails with "received 0 items of ancdata" error.
    # https://github.com/fastai/fastai/issues/23#issuecomment-345091054
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))


@broadcast_from_master
def get_random_seed():
    """Adapted from d2.utils.env:seed_all_rng()"""
    seed = os.getpid() + int(datetime.now().strftime("%S%f")) + int.from_bytes(os.urandom(2), "big")
    return seed


def setup(cfg):
    assert torch.cuda.is_available(), "cuda is not available."

    # Seed random number generators. If distributed, then sync the random seed over all GPUs.
    seed = get_random_seed()
    seed_all_rng(seed)

    LOG.info("Working Directory: {}".format(os.getcwd()))
    LOG.info("Full config:\n{}".format(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2)))

    # Set up EventStorage
    storage = WandbEventStorage()
    _CURRENT_STORAGE_STACK.append(storage)

    # After this, the cfg is immutable.
    OmegaConf.set_readonly(cfg, True)
