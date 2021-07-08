# Copyright 2021 Toyota Research Institute.  All rights reserved.
from torch.utils.data.sampler import Sampler

from detectron2.utils import comm


class InferenceGroupSampler(Sampler):
    """
    Assumptions:
        1) The dataset consists of in-order groups, i.e. [*group-1-items, *group-2-items, ...]
        2) In the dataloader, per-gpu batch size (i.e. total_batch_size / world_size) must be
           a multiple of the group size. CAVEAT: this may cause CUDA OOM.
    """
    def __init__(self, total_size, group_size):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
        """
        assert total_size > 0 and group_size > 0
        assert total_size % group_size == 0, \
           f"The total size must be divisible by group size: total size={total_size}, group size={group_size}"

        self._total_size = total_size
        self._group_size = group_size
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        self._num_groups = total_size // group_size

        shard_size = ((self._num_groups - 1) // self._world_size + 1) * self._group_size

        # shard_size = (self._total_size - 1) // self._world_size + 1
        begin = shard_size * self._rank
        end = min(shard_size * (self._rank + 1), self._total_size)
        self._local_indices = range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)
