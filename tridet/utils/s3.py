# Copyright 2021 Toyota Research Institute.  All rights reserved.
import logging
import os
import subprocess
import tempfile
import time

import requests
from hydra.utils import to_absolute_path
from tqdm import tqdm

import wandb
from detectron2.utils import comm

from tridet.utils.comm import broadcast_from_master
from tridet.utils.wandb import wandb_is_initialized

LOG = logging.getLogger(__name__)


@broadcast_from_master
def maybe_download_ckpt_from_url(cfg):
    """If the checkpoint is an S3 or https path, the main process download the weight under, by default, `/tmp/`.

    NOTE: All workers must update `cfg.MODEL.CKPT` to use the new path.
    """
    ckpt_path = cfg.MODEL.CKPT

    if ckpt_path.startswith("s3://") or ckpt_path.startswith("https://"):
        os.makedirs(cfg.TMP_DIR, exist_ok=True)
        _, ext = os.path.splitext(ckpt_path)
        tmp_path = tempfile.NamedTemporaryFile(dir=cfg.TMP_DIR, suffix=ext).name

        LOG.info("Downloading initial weights:")
        LOG.info(f"  src: {ckpt_path}")
        LOG.info(f"  dst: {tmp_path}")

        if ckpt_path.startswith("s3://"):
            if not aws_credential_is_available():
                raise ValueError('AWS credentials are undefined in environment variables.')
            s3_copy(ckpt_path, tmp_path)
        else:  # https://
            req = requests.get(ckpt_path)
            with open(tmp_path, 'wb') as f:
                for chunk in tqdm(req.iter_content(100000)):
                    f.write(chunk)
        return tmp_path

    else:
        return ckpt_path


def aws_credential_is_available():
    AWS_CREDENTIALS = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    for x in AWS_CREDENTIALS:
        if not os.environ.get(x, None):
            return False
    return True


def s3_copy(source_path, target_path, verbose=True):
    """Copy single file from local to s3, s3 to local, or s3 to s3.

    Parameters
    ----------
    source_path: str
        Path of file to copy

    target_path: str
        Path to copy file to

    verbose: bool, default: True
        If True print some helpful messages

    Returns
    -------
    bool: True if successful
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    success = False
    command_str = "aws s3 cp --acl bucket-owner-full-control {} {}".format(source_path, target_path)
    try:
        subprocess.check_output(command_str, shell=True)
        success = True
    except subprocess.CalledProcessError as e:
        success = False
        LOG.error("{} failed with error code {}".format(command_str, e.returncode))
        LOG.error(e.output)
    if verbose:
        LOG.info("Done copying file")

    return success


def sync_dir(source, target, verbose=True, excludes=None):
    """
    Sync a directory from source to target (either local to s3, s3 to s3, s3 to local)

    Parameters
    ----------
    source: str
        Directory from which we want to sync files

    target: str
        Directory to which all files will be synced

    verbose: bool, default: True
        If True, log some helpful messages
    """
    assert source.startswith('s3://') or target.startswith('s3://')
    command_str = "aws s3 sync --quiet --acl bucket-owner-full-control {} {}".format(source, target)
    if excludes:
        for exclude in excludes:
            command_str += f" --exclude '{exclude}'"
    if verbose:
        LOG.info("Syncing with '{}'".format(command_str))
    try:
        subprocess.check_output(command_str, shell=True)
    except subprocess.CalledProcessError as e:
        LOG.error("{} failed with error code {}".format(command_str, e.returncode))
        LOG.error(e.output)
    if verbose:
        LOG.info("Done syncing")


def sync_output_dir_s3(cfg, output_dir=None):
    output_dir = output_dir or os.getcwd()
    output_dir = os.path.abspath(os.path.normpath(output_dir))
    output_root = to_absolute_path(cfg.OUTPUT_ROOT)

    assert os.path.commonprefix([output_dir, output_root]) == output_root, f'{output_dir}, {output_root}'
    tar_output_dir = os.path.join(cfg.SYNC_OUTPUT_DIR_S3.ROOT_IN_S3, output_dir[len(output_root) + 1:])

    if comm.is_main_process():
        LOG.info(f"Syncing output_dir: {output_dir} -> {tar_output_dir}")
        sync_dir(output_dir, tar_output_dir)

        if wandb_is_initialized():
            tar_wandb_run_dir = os.path.join(tar_output_dir, 'wandb')
            LOG.info(f"Syncing W&B run dir: {wandb.run.dir} -> {tar_wandb_run_dir}")
            sync_dir(wandb.run.dir, tar_wandb_run_dir)

    elif comm.get_local_rank() == 0 and os.path.exists(os.path.join(output_dir, 'logs')):
        # local master -- only sync the log files
        log_output_dir, log_tar_output_dir = os.path.join(output_dir, 'logs'), os.path.join(tar_output_dir, 'logs')
        LOG.info(f"Syncing log output_dir: {log_output_dir} -> {log_tar_output_dir}")
        sync_dir(log_output_dir, log_tar_output_dir)


def maybe_download_from_s3(src_path):
    if not src_path.startswith("s3://"):
        return src_path

    extension = os.path.splitext(src_path)[-1]
    if not extension:
        extension = None
    tmp_path = tempfile.NamedTemporaryFile(suffix=extension).name
    suceeded = s3_copy(src_path, tmp_path)
    if not suceeded:
        raise RuntimeError("`s3_copy` failed.")
    return tmp_path


def maybe_sync_dir_from_s3(src_path, excludes=None):
    if not src_path.startswith("s3://"):
        return src_path

    tmp_dir = tempfile.NamedTemporaryFile().name
    os.makedirs(tmp_dir)
    LOG.info(f"Syncing {src_path} to {tmp_dir}")
    st = time.time()
    sync_dir(src_path, tmp_dir, excludes=excludes)
    LOG.info(f"Done. ({time.time() - st}s)")
    return tmp_dir
