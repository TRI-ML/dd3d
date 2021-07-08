#!/usr/bin/env python
# Copyright 2021 Toyota Research Institute.  All rights reserved.
import logging
import os
from collections import OrderedDict, defaultdict

import hydra
import torch
import wandb
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
from torch.cuda import amp
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import detectron2.utils.comm as d2_comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluators, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import CommonMetricPrinter, get_event_storage

import tridet.modeling  # pylint: disable=unused-import
import tridet.utils.comm as comm
from tridet.data import build_test_dataloader, build_train_dataloader
from tridet.data.dataset_mappers import get_dataset_mapper
from tridet.data.datasets import random_sample_dataset_dicts, register_datasets
from tridet.evaluators import get_evaluator
from tridet.modeling import build_tta_model
from tridet.utils.s3 import sync_output_dir_s3
from tridet.utils.setup import setup
from tridet.utils.train import get_inference_output_dir, print_test_results
from tridet.utils.visualization import mosaic, save_vis
from tridet.utils.wandb import flatten_dict, log_nested_dict
from tridet.visualizers import get_dataloader_visualizer, get_predictions_visualizer

LOG = logging.getLogger('tridet')


@hydra.main(config_path="../configs/", config_name="defaults")
def main(cfg):
    setup(cfg)
    dataset_names = register_datasets(cfg)
    if cfg.ONLY_REGISTER_DATASETS:
        return {}, cfg
    LOG.info(f"Registered {len(dataset_names)} datasets:" + '\n\t' + '\n\t'.join(dataset_names))

    model = build_model(cfg)

    checkpoint_file = cfg.MODEL.CKPT
    if checkpoint_file:
        Checkpointer(model).load(checkpoint_file)

    if cfg.EVAL_ONLY:
        assert cfg.TEST.ENABLED, "'eval-only' mode is not compatible with 'cfg.TEST.ENABLED = False'."
        test_results = do_test(cfg, model, is_last=True)
        if cfg.TEST.AUG.ENABLED:
            test_results.update(do_test(cfg, model, is_last=True, use_tta=True))
        return test_results, cfg

    if comm.is_distributed():
        assert d2_comm._LOCAL_PROCESS_GROUP is not None
        # Convert all Batchnorm*D to nn.SyncBatchNorm.
        # For faster training, the batch stats are computed over only the GPUs of the same machines (usually 8).
        sync_bn_pg = d2_comm._LOCAL_PROCESS_GROUP if cfg.SOLVER.SYNCBN_USE_LOCAL_WORKERS else None
        model = SyncBatchNorm.convert_sync_batchnorm(model, process_group=sync_bn_pg)
        model = DistributedDataParallel(
            model,
            device_ids=[d2_comm.get_local_rank()],
            broadcast_buffers=False,
            find_unused_parameters=cfg.SOLVER.DDP_FIND_UNUSED_PARAMETERS
        )

    do_train(cfg, model)
    test_results = do_test(cfg, model, is_last=True)
    if cfg.TEST.AUG.ENABLED:
        test_results.update(do_test(cfg, model, is_last=True, use_tta=True))
    return test_results, cfg


def do_train(cfg, model):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = Checkpointer(model, './', optimizer=optimizer, scheduler=scheduler)
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)

    writers = [CommonMetricPrinter(max_iter)] if d2_comm.is_main_process() else []

    train_mapper = get_dataset_mapper(cfg, is_train=True)
    dataloader, dataset_dicts = build_train_dataloader(cfg, mapper=train_mapper)
    LOG.info("Length of train dataset: {:d}".format(len(dataset_dicts)))
    LOG.info("Starting training")
    storage = get_event_storage()

    if cfg.EVAL_ON_START:
        do_test(cfg, model)
        comm.synchronize()

    # In mixed-precision training, gradients are scaled up to keep them from being vanished due to half-precision.
    # They're scaled down again before optimizers use them to compute updates.
    scaler = amp.GradScaler(enabled=cfg.SOLVER.MIXED_PRECISION_ENABLED)

    # Accumulate gradients for multiple batches (as returned by dataloader) before calling optimizer.step().
    accumulate_grad_batches = cfg.SOLVER.ACCUMULATE_GRAD_BATCHES

    num_images_seen = 0
    # For logging, this stores losses aggregated from all workers in distributed training.
    batch_loss_dict = defaultdict(float)
    optimizer.zero_grad()
    for data, iteration in zip(dataloader, range(max_iter * accumulate_grad_batches)):
        iteration += 1
        # this assumes drop_last=True, so all workers has the same size of batch.
        num_images_seen += len(data) * d2_comm.get_world_size()
        if iteration % accumulate_grad_batches == 0:
            storage.step()

        with amp.autocast(enabled=cfg.SOLVER.MIXED_PRECISION_ENABLED):
            loss_dict = model(data)
        # Account for accumulated gradients.
        loss_dict = {name: loss / accumulate_grad_batches for name, loss in loss_dict.items()}
        losses = sum(loss_dict.values())
        # FIXME: First few iterations might give Inf/NaN losses when using mixed precision. What should be done?
        if not torch.isfinite(losses):
            LOG.critical(f"The loss DIVERGED: {loss_dict}")

        # Track total loss for logging.
        loss_dict_reduced = {k: v.item() for k, v in d2_comm.reduce_dict(loss_dict).items()}
        assert torch.isfinite(torch.as_tensor(list(loss_dict_reduced.values()))).all(), loss_dict_reduced
        for k, v in loss_dict_reduced.items():
            batch_loss_dict[k] += v

        # No amp version: leaving this here for legacy:
        # losses.backward()
        scaler.scale(losses).backward()

        if iteration % accumulate_grad_batches > 0:
            # Just accumulate gradients and move on to next batch.
            continue

        scaler.step(optimizer)
        storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
        scheduler.step()
        scaler.update()

        losses_reduced = sum(loss for loss in batch_loss_dict.values())
        storage.put_scalars(total_loss=losses_reduced, **batch_loss_dict)

        # Reset states.
        batch_loss_dict = defaultdict(float)
        optimizer.zero_grad()

        batch_iter = iteration // accumulate_grad_batches

        # TODO: probably check if the gradients contain any inf or nan, and only proceed if not.
        if batch_iter > 5 and (batch_iter % 20 == 0 or batch_iter == max_iter):
            # if batch_iter > -1 and (batch_iter % 1 == 0 or batch_iter == max_iter):
            for writer in writers:
                writer.write()
            # log epoch, # images seen
            if d2_comm.is_main_process() and cfg.WANDB.ENABLED:
                wandb.log({"epoch": 1 + num_images_seen // len(dataset_dicts)}, step=batch_iter)
                wandb.log({"num_images_seen": num_images_seen}, step=batch_iter)

        if cfg.VIS.DATALOADER_ENABLED and batch_iter % cfg.VIS.DATALOADER_PERIOD == 0 and d2_comm.is_main_process():
            dataset_name = cfg.DATASETS.TRAIN.NAME
            visualizer_names = MetadataCatalog.get(dataset_name).loader_visualizers
            viz_images = defaultdict(dict)
            for viz_name in visualizer_names:
                viz = get_dataloader_visualizer(cfg, viz_name, dataset_name)
                for idx, x in enumerate(data):
                    viz_images[idx].update(viz.visualize(x))

            if cfg.WANDB.ENABLED:
                per_image_vis = [mosaic(list(viz_images[idx].values())) for idx in range(len(data))]
                wandb.log({
                    "dataloader": [wandb.Image(vis, caption=f"idx={idx}") for idx, vis in enumerate(per_image_vis)]
                },
                          step=batch_iter)
            save_vis(viz_images, os.path.join(os.getcwd(), "visualization"), "dataloader", step=batch_iter)

        if d2_comm.is_main_process():
            periodic_checkpointer.step(batch_iter - 1)  # (fvcore) model_0004999.pth checkpoints 5000-th iteration

        if cfg.SYNC_OUTPUT_DIR_S3.ENABLED and batch_iter > 0 and batch_iter % cfg.SYNC_OUTPUT_DIR_S3.PERIOD == 0:
            sync_output_dir_s3(cfg)

        if (cfg.TEST.EVAL_PERIOD > 0 and batch_iter % cfg.TEST.EVAL_PERIOD == 0 and batch_iter != max_iter) or \
            batch_iter in cfg.TEST.ADDITIONAL_EVAL_STEPS:
            do_test(cfg, model)
            d2_comm.synchronize()


def do_test(cfg, model, is_last=False, use_tta=False):
    if not cfg.TEST.ENABLED:
        LOG.warning("Test is disabled.")
        return {}

    dataset_names = [cfg.DATASETS.TEST.NAME]  # NOTE: only support single test dataset for now.

    if use_tta:
        LOG.info("Starting inference with test-time augmentation.")
        if isinstance(model, DistributedDataParallel):
            model.module.postprocess_in_inference = False
        else:
            model.postprocess_in_inference = False
        model = build_tta_model(cfg, model)

    test_results = OrderedDict()
    for dataset_name in dataset_names:
        # output directory for this dataset.
        dset_output_dir = get_inference_output_dir(dataset_name, is_last=is_last, use_tta=use_tta)

        # What evaluators are used for this dataset?
        evaluator_names = MetadataCatalog.get(dataset_name).evaluators
        evaluators = []
        for evaluator_name in evaluator_names:
            evaluator = get_evaluator(cfg, dataset_name, evaluator_name, dset_output_dir)
            evaluators.append(evaluator)
        evaluator = DatasetEvaluators(evaluators)

        mapper = get_dataset_mapper(cfg, is_train=False)
        dataloader, dataset_dicts = build_test_dataloader(cfg, dataset_name, mapper)

        per_dataset_results = inference_on_dataset(model, dataloader, evaluator)
        if use_tta:
            per_dataset_results = OrderedDict({k + '-tta': v for k, v in per_dataset_results.items()})
        test_results[dataset_name] = per_dataset_results

        if cfg.VIS.PREDICTIONS_ENABLED and d2_comm.is_main_process():
            visualizer_names = MetadataCatalog.get(dataset_name).pred_visualizers
            # Randomly (but deterministically) select what samples to visualize.
            # The samples are shared across all visualizers and iterations.
            sampled_dataset_dicts, inds = random_sample_dataset_dicts(
                dataset_name, num_samples=cfg.VIS.PREDICTIONS_MAX_NUM_SAMPLES
            )

            viz_images = defaultdict(dict)
            for viz_name in visualizer_names:
                LOG.info(f"Running prediction visualizer: {viz_name}")
                visualizer = get_predictions_visualizer(cfg, viz_name, dataset_name, dset_output_dir)
                for x in tqdm(sampled_dataset_dicts):
                    sample_id = x['sample_id']
                    viz_images[sample_id].update(visualizer.visualize(x))

            save_vis(viz_images, dset_output_dir, "visualization")

            if cfg.WANDB.ENABLED:
                LOG.info(f"Uploading prediction visualization to W&B: {dataset_name}")
                for sample_id in viz_images.keys():
                    viz_images[sample_id] = mosaic(list(viz_images[sample_id].values()))
                step = get_event_storage().iter
                wandb.log({
                    f"{dataset_name}-predictions":
                    [wandb.Image(viz, caption=f"{sample_id}") for sample_id, viz in viz_images.items()]
                },
                          step=step)

    test_results = flatten_dict(test_results)
    log_nested_dict(test_results)
    if d2_comm.is_main_process():
        LOG.info("Evaluation results for {} in csv format:".format(dataset_name))
        print_test_results(test_results)

    if use_tta:
        if isinstance(model, DistributedDataParallel):
            model.module.postprocess_in_inference = True
        else:
            model.postprocess_in_inference = True

    return test_results


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
    LOG.info("DONE.")
