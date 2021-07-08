import logging
import os

from hydra.experimental.callback import Callback
from mpi4py import MPI

from detectron2.utils import comm as d2_comm
from detectron2.utils.logger import setup_logger

from tridet.utils.s3 import aws_credential_is_available, maybe_download_ckpt_from_url, sync_output_dir_s3
from tridet.utils.setup import setup_distributed
from tridet.utils.wandb import derive_output_dir_from_wandb_id, init_wandb, wandb_credential_is_available

LOG = logging.getLogger(__name__)


class SetupDistributedCallback(Callback):
    """
    """
    def on_run_start(self, config, **kwargs):  # pylint: disable=unused-argument
        world_size = MPI.COMM_WORLD.Get_size()
        distributed = world_size > 1
        if distributed:
            rank = MPI.COMM_WORLD.Get_rank()
            setup_distributed(world_size, rank)

    def on_job_start(self, config, **kwargs):  # pylint: disable=unused-argument
        world_size = d2_comm.get_world_size()
        rank = d2_comm.get_rank()
        LOG.info("Rank of current process: {}. World size: {}".format(rank, world_size))


class WandbInitCallback(Callback):
    """If W&B is enabled, then
        1) initialize W&B,
        2) derive the path of output directory using W&B ID, and
        3) set it as hydra working directory.
    """
    def on_run_start(self, config, **kwargs):  # pylint: disable=unused-argument
        if not config.WANDB.ENABLED:
            return
        if not wandb_credential_is_available():
            LOG.warning(
                "W&B credential must be defined in environment variables."
                "Use `WANDB.ENABLED=False` to suppress this warning. "
                "Skipping `WandbInitCallback`..."
            )
            return

        init_wandb(config)
        output_dir = derive_output_dir_from_wandb_id(config)
        if output_dir:
            config.hydra.run.dir = output_dir


class SyncOutputDirCallback(Callback):
    def on_run_start(self, config, **kwargs):  # pylint: disable=unused-argument
        if d2_comm.is_main_process():
            output_dir = config.hydra.run.dir
        else:
            output_dir = None
        output_dir = MPI.COMM_WORLD.bcast(output_dir, root=0)

        if output_dir != config.hydra.run.dir:
            LOG.warning("Hydra run dir is not synced. Overwriting from rank=0.")
            config.hydra.run.dir = output_dir


class D2LoggerCallback(Callback):
    def on_run_start(self, config, **kwargs):  # pylint: disable=unused-argument
        rank = d2_comm.get_rank()
        log_output_dir = os.path.join(config.hydra.run.dir, 'logs')
        setup_logger(log_output_dir, distributed_rank=rank, name="hydra")
        setup_logger(log_output_dir, distributed_rank=rank, name="detectron2", abbrev_name="d2")
        setup_logger(log_output_dir, distributed_rank=rank, name="tridet")
        setup_logger(log_output_dir, distributed_rank=rank, name="fvcore")

        logging.getLogger('numba').setLevel(logging.ERROR)  # too much logs


class CkptPathResolverCallback(Callback):
    """
    If the checkpoint (`config.model.CKPT`) is an S3 path, then downloaded it and replace the path with
    local path.
    """
    def on_run_start(self, config, **kwargs):  # pylint: disable=unused-argument
        if config.MODEL.CKPT:
            new_ckpt_path = maybe_download_ckpt_from_url(config)
            new_ckpt_path = os.path.abspath(new_ckpt_path)
            config.MODEL.CKPT = new_ckpt_path


class SyncOutputS3BeforeEnd(Callback):
    """
    """
    def on_run_start(self, config, **kwargs):  # pylint: disable=unused-argument
        if config.SYNC_OUTPUT_DIR_S3.ENABLED and not aws_credential_is_available():
            raise ValueError(f"\n\nAWS credential must be set in environment variables (rank={d2_comm.get_rank()}).\n")

    def on_run_end(self, config, **kwargs):  # pylint: disable=unused-argument
        """
        """
        if config.SYNC_OUTPUT_DIR_S3.ENABLED:
            sync_output_dir_s3(config, output_dir=config.hydra.run.dir)
