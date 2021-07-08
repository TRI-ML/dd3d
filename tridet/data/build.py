# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2021 Toyota Research Institute.  All rights reserved.
import itertools
import logging
import operator
import time

import numpy as np
import torch
from mpi4py import MPI
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from detectron2.data.build import filter_images_with_only_crowd_annotations, print_instances_class_histogram
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.common import AspectRatioGroupedDataset, DatasetFromList, MapDataset
from detectron2.data.samplers import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler
from detectron2.utils import comm
from detectron2.utils.env import seed_all_rng

from tridet.data.samplers import InferenceGroupSampler
from tridet.utils.comm import is_distributed
from tridet.utils.tasks import TaskManager

LOG = logging.getLogger(__name__)


def build_train_dataloader(cfg, mapper):
    train_dataset_name = cfg.DATASETS.TRAIN.NAME
    dataset_dicts = DatasetCatalog.get(train_dataset_name)

    tm = TaskManager(cfg)
    if tm.has_detection_task:
        if cfg.DATALOADER.TRAIN.FILTER_EMPTY_ANNOTATIONS:
            dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)

        class_names = MetadataCatalog.get(train_dataset_name).thing_classes
        print_instances_class_histogram(dataset_dicts, class_names)

    dataset = DatasetFromList(dataset_dicts, copy=False)
    dataset = MapDataset(dataset, mapper)

    # Sampler
    sampler_name = cfg.DATALOADER.TRAIN.SAMPLER
    LOG.info("Using training sampler {}".format(sampler_name))
    # TODO avoid if-else?
    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts, cfg.DATALOADER.TRAIN.REPEAT_THRESHOLD
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    data_loader = build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.TRAIN.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.TRAIN.NUM_WORKERS,
    )
    return data_loader, dataset_dicts


def build_test_dataloader(cfg, dataset_name, mapper):
    dataset_dicts = DatasetCatalog.get(dataset_name)
    assert len(dataset_dicts), "Dataset '{}' is empty!".format(dataset_name)

    dataset = DatasetFromList(dataset_dicts, copy=False)
    dataset = MapDataset(dataset, mapper)

    # Sampler
    sampler_name = cfg.DATALOADER.TEST.SAMPLER
    LOG.info("Using test sampler {}".format(sampler_name))
    if sampler_name == "InferenceSampler":
        sampler = InferenceSampler(len(dataset))
    elif sampler_name == "InferenceGroupSampler":
        assert cfg.TEST.IMS_PER_BATCH % (comm.get_world_size() * cfg.DATALOADER.TEST.NUM_IMAGES_PER_GROUP) == 0
        sampler = InferenceGroupSampler(len(dataset), cfg.DATALOADER.TEST.NUM_IMAGES_PER_GROUP)
    else:
        raise ValueError(f'Invalid test sampler name: {sampler_name}')
    # For benchmarking inference time, use 1 image per worker.
    # This is the standard when reporting inference time in papers.
    world_size = comm.get_world_size()
    total_batch_size = cfg.TEST.IMS_PER_BATCH
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(total_batch_size, world_size)
    batch_size = total_batch_size // world_size
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.TEST.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        pin_memory=True
    )
    assert len(data_loader.dataset) == len(dataset_dicts)
    return data_loader, dataset_dicts


def build_batch_data_loader(dataset, sampler, total_batch_size, *, aspect_ratio_grouping=False, num_workers=0):
    """
    Build a batched dataloader for training.

    Args:
        dataset (torch.utils.data.Dataset): map-style PyTorch dataset. Can be indexed.
        sampler (torch.utils.data.sampler.Sampler): a sampler that produces indices
        total_batch_size (int): total batch size across GPUs.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    world_size = comm.get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(total_batch_size, world_size)

    batch_size = total_batch_size // world_size
    if aspect_ratio_grouping:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        return AspectRatioGroupedDataset(data_loader, batch_size)
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last=True
        )  # drop_last so the batch always have the same size
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
            worker_init_fn=worker_init_reset_seed,
        )


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2**31) + worker_id)


def collect_dataset_dicts(d2_dataset, num_workers_per_gpu=8, dummy_batch_size=32):
    """Build D2 dataset (i.e. List[Dict]), given a dataset implementing recipe for building its item.

    This is useful when a __getitem__() takes much time / memory, so it's desirable to do one-time
    conversion into `List[Dict]` format. This function repurpose multi-node distributed training tools
    of pytorch to acccelerate data loading.

    NOTE: Both the input dataset object and the resulting D2 dataset, multiplied by # of GPUs, must fit in memory.

    Parameters
    ----------
    d2_dataset: Dataset
        __getitem__() returns a D2-formatted dictionary.
    """
    class TrivialModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = nn.Parameter(torch.FloatTensor([0.]))

        def forward(self, x):  # pylint: disable=unused-argument
            return self.theta * 0.

    model = TrivialModel().to('cuda')

    if is_distributed():
        model = DistributedDataParallel(
            model,
            device_ids=[comm.get_local_rank()],
            broadcast_buffers=False,
        )
        sampler = InferenceSampler(len(d2_dataset))  # d2 dependency
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        d2_dataset,
        num_workers=num_workers_per_gpu,
        sampler=sampler,
        batch_size=dummy_batch_size,
        collate_fn=lambda x: x,
        drop_last=False,
    )
    d2_dicts = []
    LOG.info(f"Creating D2 dataset: {len(dataloader)} batches on rank {comm.get_rank()}.")
    for x in tqdm(dataloader):
        loss = model(x)
        loss.backward()
        d2_dicts.extend(x)

    LOG.info("Gathering D2 dataset dicts from all GPU workers...")
    st = time.time()
    d2_dicts = MPI.COMM_WORLD.allgather(d2_dicts)
    took = time.time() - st
    d2_dicts = list(itertools.chain.from_iterable(d2_dicts))
    LOG.info(f"Done (length={len(d2_dicts)}, took={took:.1f}s).")

    return d2_dicts
