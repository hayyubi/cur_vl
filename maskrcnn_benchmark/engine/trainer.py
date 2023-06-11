# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import os
import time
import collections
import copy
from collections import defaultdict

import torch
import torch.distributed as dist
from tqdm import tqdm

from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.utils.comm import get_world_size, synchronize
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector.mmss_rcnn import MMSSRegionModel
from maskrcnn_benchmark.engine.inference import phased_inference
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

from apex import amp

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def write_tf_summary(dictionary, tb_logger, iteration, prefix=''):
    if get_world_size() > 1: 
        if dist.get_rank() != 0 or tb_logger is None: return
    else:
        if tb_logger is None: return

    for k, v in dictionary.items():
        k2 = f'{prefix}/{k}'
        if isinstance(v, collections.Mapping):
            write_tf_summary(v, tb_logger, iteration, prefix=k2)
        else:
            tb_logger.add_scalar(k2, v, iteration)


def do_train(
    cfg,
    model,
    data_loaders_curr,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    test_period,
    arguments,
    tb_logger,
    prev_model,
    prev_model_checkpointer=None,
    data_loader_inference = None,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = 0
    for data_loader in data_loaders_curr:
        max_iter += len(data_loader)
    start_iter_curr = arguments["iteration"]

    # Take care of start_iter in case of Curriculum learning
    if cfg.CURRICULUM.DO:
        curr_iters = cfg.CURRICULUM.ITERS
        phases = cfg.CURRICULUM.NUM_PHASES
        cum_iter = 0
        cur_phase = 0
        for i, itter in enumerate(curr_iters):
            cum_iter += itter
            if start_iter_curr < cum_iter:
                cum_iter = cum_iter - itter if i!= 0 else 0
                start_iter_curr = start_iter_curr - cum_iter
                break
            cur_phase += 1
    else:
        cur_phase = 0
        phases = 1

    model.train()
    start_training_time = time.time()
    end = time.time()
    
    if prev_model:
        prev_module = prev_model.module if isinstance(prev_model, torch.nn.parallel.DistributedDataParallel) else prev_model

    # Loading Previous phase weight to the prev model
    if cur_phase > 0 and prev_model:
        last_phase_ckpt_iter = sum(cfg.CURRICULUM.ITERS[0:cur_phase])
        last_phase_checkpoint = os.path.join(prev_model_checkpointer.save_dir, "model_{:07d}.pth".format(last_phase_ckpt_iter))
        _ = prev_model_checkpointer.load(last_phase_checkpoint, use_latest=False)
        for p in prev_model.parameters():
            p.requires_grad = False
        prev_module.set_prev_knowledge_source_mode()

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    dataset_names = cfg.DATASETS.TEST

    dl_val_list = data_loader_val
    data_loader_val = data_loader_val[0]
 
    module = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    if hasattr(module, 'roi_heads') and 'box' in module.roi_heads:
        if module.roi_heads['box'].predictor.embedding_based:
            module.roi_heads['box'].predictor.set_class_embeddings(
                data_loader_val.dataset.class_emb_mtx)


    if type(module) is MMSSRegionModel:
        # Load val dataset embedding matrix, which we interested in testing against.
        emb_mtx = data_loader_val.dataset.class_emb_mtx if cfg.DATASETS.DATASET_ARGS.LOAD_EMBEDDINGS else None
        module.set_class_list(data_loader_val.dataset.contiguous_class_list)
        module.set_class_embeddings(emb_mtx)
        if prev_model:
            prev_module.set_class_list(data_loader_val.dataset.contiguous_class_list)
            prev_module.set_class_embeddings(emb_mtx)
        # Reset, in case returning from middle of training or even in the beginning
        if not module.pretrained_class_emb:
            module.need_cls_embedding_reset = 1
            module.reset_class_embeddings()
            if prev_model:
                prev_module.need_cls_embedding_reset = 1
                prev_module.reset_class_embeddings()
   
    default_gpu = False
    if get_world_size() >= 2:
        if dist.get_rank() == 0:
            default_gpu = True
    else:
        default_gpu = True

    prev_iter = 0
    prev_knowledge = None
    for phase in list(range(phases)):
        if phase < cur_phase:
            prev_iter += cfg.CURRICULUM.ITERS[phase]
            continue 
        elif phase > cur_phase:
            start_iter = 0
        else:
            start_iter = start_iter_curr

        data_loader = data_loaders_curr[phase]
        iteration = prev_iter + start_iter
        for it, (images, targets, _) in enumerate(data_loader, start_iter):

            if it == len(data_loader):
                break

            iteration = prev_iter + it

            if any(len(target) < 1 for target in targets):
                logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
                continue
            data_time = time.time() - end
            iteration = iteration + 1
            arguments["iteration"] = iteration

            images = images.to(device)
            try:
                targets = [target.to(device) for target in targets]
            except:
                pass

            # For previous model alignment curriculum get previous knowledge
            if prev_model and prev_module.prev_knowledge_source_mode:
                prev_knowledge = prev_model(images, targets)

            # with torch.autograd.detect_anomaly():
            loss_dict = model(images, targets, iteration/max_iter, prev_knowledge)

            if isinstance(loss_dict, tuple):
                info_dict, loss_dict = loss_dict
            else:
                info_dict = None

            losses = sum(loss for loss in loss_dict.values())
            losses = losses / float(cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss_dict_reduced['loss'] = losses_reduced
            meters.update(**loss_dict_reduced)

            if info_dict is not None:
                info_dict_reduced = reduce_loss_dict(info_dict)
                meters.update(**info_dict_reduced)

            # Note: If mixed precision is not used, this ends up doing nothing
            # Otherwise apply loss scaling for mixed-precision recipe
            with amp.scale_loss(losses, optimizer) as scaled_losses:
                scaled_losses.backward()

            if iteration % cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS == 0:
                if cfg.SOLVER.CLIP_GRAD_NORM_AT > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.SOLVER.CLIP_GRAD_NORM_AT)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % cfg.SOLVER.LOG_PERIOD == 0 or iteration == max_iter:
                write_tf_summary(loss_dict_reduced, tb_logger, iteration, prefix='train')
                if info_dict is not None:
                    write_tf_summary(info_dict_reduced, tb_logger, iteration, prefix='train')

                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
            if iteration % checkpoint_period == 0:
                checkpointer.save("model_{:07d}".format(iteration), **arguments)
            if test_period > 0 and iteration % test_period == 0:
                meters_val = MetricLogger(delimiter="  ")
                synchronize()
                if cfg.TEST.DO_EVAL:
                    for dl, dname in zip(dl_val_list, cfg.DATASETS.TEST):
                        val_results = inference(
                            model,
                            dl,
                            dataset_name=dname,
                            iou_types=iou_types,
                            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                            device=cfg.MODEL.DEVICE,
                            expected_results=cfg.TEST.EXPECTED_RESULTS,
                            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                            output_folder=None,
                        )
                        synchronize()

                        if val_results is not None:
                            val_results, _ = val_results
                            val_results = val_results.results
                            write_tf_summary(val_results, tb_logger, iteration, prefix=f'validation/{dname}')

                    if hasattr(module, 'roi_heads') and 'box' in module.roi_heads:
                        if module.roi_heads['box'].predictor.embedding_based:
                            module.roi_heads['box'].predictor.set_class_embeddings(
                                data_loader.dataset.class_emb_mtx)

                if not cfg.SOLVER.SKIP_VAL_LOSS:
                    with torch.no_grad():
                        if cfg.SOLVER.USE_TRAIN_MODE_FOR_VALIDATION_LOSS:
                            model.train()
                        else:
                            model.eval()
                        for iteration_val, (images_val, targets_val, _) in enumerate(tqdm(data_loader_val)):
                            images_val = images_val.to(device)
                            try:
                                targets_val = [target.to(device) for target in targets_val]
                            except:
                                pass
                            loss_dict = model(images_val, targets_val, iteration/max_iter)
                            if isinstance(loss_dict, tuple):
                                info_dict, loss_dict = loss_dict
                            else:
                                info_dict = None
                            losses = sum(loss for loss in loss_dict.values())
                            loss_dict_reduced = reduce_loss_dict(loss_dict)
                            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                            meters_val.update(loss=losses_reduced, **loss_dict_reduced)
                            if info_dict is not None:
                                info_dict_reduced = reduce_loss_dict(info_dict)
                                meters_val.update(**info_dict_reduced)
                    logger.info(
                        meters_val.delimiter.join(
                            [
                                "[Validation]: ",
                                "eta: {eta}",
                                "iter: {iter}",
                                "{meters}",
                                "lr: {lr:.6f}",
                                "max mem: {memory:.0f}",
                            ]
                        ).format(
                            eta=eta_string,
                            iter=iteration,
                            meters=str(meters_val),
                            lr=optimizer.param_groups[0]["lr"],
                            memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        )
                    )
                    meters_val_dict = {k: v.global_avg for k, v in meters_val.meters.items()}
                    write_tf_summary(meters_val_dict, tb_logger, iteration,
                                     prefix=f'validation/{cfg.DATASETS.TEST[0]}')
                model.train()
                synchronize()
            
        prev_iter = iteration
        if prev_model:
            prev_model = do_previous_model_alignment_curriculum_logistics(model, prev_model)

        # Save checkpoint at the end of each phase
        checkpointer.save("model_{:07d}".format(iteration), **arguments)

        # Calculate accuracy of objects by phases of introduction
        if default_gpu:
            for dl, dname in zip(data_loader_inference, cfg.DATASETS.TEST):
                if hasattr(dl.dataset, 'objects_by_phases') :
                    # Load embedding matrix since it may differ by datasets
                    if type(module) is MMSSRegionModel:
                        emb_mtx = dl.dataset.class_emb_mtx if cfg.DATASETS.DATASET_ARGS.LOAD_EMBEDDINGS else None
                        module.set_class_list(dl.dataset.contiguous_class_list)
                        module.set_class_embeddings(emb_mtx)
                        if not module.pretrained_class_emb:
                            module.need_cls_embedding_reset = 1
                            module.reset_class_embeddings()

                    objects_by_phases = dl.dataset.objects_by_phases
                    inference_out_dict = phased_inference(model,
                                                    dl,
                                                    objects_by_phases,
                                                    device,
                                                    cfg.TEST.PHASED_INFERENCE)
                    logger.info(
                        "   ".join(
                            [
                                "[Inference/{dset}]: ",
                                "End of Phase: {phase}",
                                "iter: {iter}",
                                "{meters}",
                            ]
                        ).format(
                            dset=dname,
                            phase=phase+1,
                            iter=iteration,
                            meters=str(inference_out_dict),
                        )
                    )
                    write_tf_summary(inference_out_dict, tb_logger, iteration,
                                     prefix=f'validation/{dname}')
                    module.set_phased_inference(False)
                    model.train()

            # Reset embedding matrix after inference, since inference may contain other datasets
            if type(module) is MMSSRegionModel:
                emb_mtx = data_loader_val.dataset.class_emb_mtx if cfg.DATASETS.DATASET_ARGS.LOAD_EMBEDDINGS else None
                module.set_class_list(data_loader_val.dataset.contiguous_class_list)
                module.set_class_embeddings(emb_mtx)
                if not module.pretrained_class_emb:
                    module.need_cls_embedding_reset = 1
                    module.reset_class_embeddings()


        synchronize()

        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

def do_previous_model_alignment_curriculum_logistics(model, prev_model):
    prev_model.load_state_dict(copy.deepcopy(model.state_dict()))
    for p in prev_model.parameters():
        p.requires_grad = False
    
    prev_module = prev_model.module if isinstance(prev_model, torch.nn.parallel.DistributedDataParallel)\
                     else prev_model
    prev_module.set_prev_knowledge_source_mode()

    return prev_model