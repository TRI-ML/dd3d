# Copyright 2021 Toyota Research Institute.  All rights reserved.
import logging
import os

from tabulate import tabulate
from termcolor import colored

from detectron2.utils.events import get_event_storage

LOG = logging.getLogger(__name__)


def get_inference_output_dir(dataset_name, is_last=False, use_tta=False, root_output_dir=None):
    if not root_output_dir:
        root_output_dir = os.getcwd()  # hydra
    step = get_event_storage().iter
    if is_last:
        result_dirname = "final"
    else:
        result_dirname = f"step{step:07d}"
    if use_tta:
        result_dirname += "-tta"
    output_dir = os.path.join(root_output_dir, "inference", result_dirname, dataset_name)
    return output_dir


def print_test_results(test_results):
    metric_table = tabulate(
        [(k, v) for k, v in test_results.items()],
        headers=["metric", "value"],
        tablefmt="pipe",
        numalign="left",
        stralign="left",
    )
    LOG.info("Test results:\n" + colored(metric_table, "cyan"))
