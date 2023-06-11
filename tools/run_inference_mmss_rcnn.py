# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import json

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.timer import Timer, get_time_str
from tqdm import tqdm

# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    logger = setup_logger("maskrcnn_benchmark", output_dir, 
                            get_rank(), filename="inference_log.txt")
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)

    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16'
    amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)

    # Set inference from image for datasets so mentioned in config
    inference_from_image_list = [False for _ in range(len(dataset_names))]
    for k in cfg.DATASETS.DATASET_ARGS.INFERENCE_FROM_IMAGE:
        inference_from_image_list[k] = True

    model.eval()
    device = torch.device(cfg.MODEL.DEVICE)
    for i, (output_folder, dataset_name, data_loader_val) in enumerate(zip(output_folders, dataset_names, data_loaders_val)):
        # Model's class embedding depends on dataset
        module = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        emb_mtx = data_loader_val.dataset.class_emb_mtx if cfg.DATASETS.DATASET_ARGS.LOAD_EMBEDDINGS else None
        module.set_class_list(data_loader_val.dataset.contiguous_class_list)
        module.set_class_embeddings(emb_mtx)

        if inference_from_image_list[i]:
            module.set_inference_from_image_true()

        meters_val = MetricLogger(delimiter="  ")
        timer = Timer()
        with torch.no_grad():
            timer.tic()
            for iteration_val, (images_val, targets_val, _) in enumerate(tqdm(data_loader_val)):
                images_val = images_val.to(device)
                try:
                    targets_val = [target.to(device) for target in targets_val]
                except:
                    pass
                result_dict = model(images_val, targets_val)
                result_dict_reduced = reduce_loss_dict(result_dict)
                meters_val.update(**result_dict)
        synchronize()
        time = timer.toc()
        time_str = get_time_str(time)

        logger.info(
            meters_val.delimiter.join(
                [
                    "[{dataset}]: ",
                    "Inference time: {time}",
                    "{meters}",
                ]
            ).format(
                dataset=dataset_name,
                time=time_str,
                meters=str(meters_val),
            )
        )
        meters_val_dict = {k: v.global_avg for k, v in meters_val.meters.items()}
        with open(os.path.join(output_folder, 'result.json'), "w") as f:
            json.dump(meters_val_dict, f)


if __name__ == "__main__":
    main()
