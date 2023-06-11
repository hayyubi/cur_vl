# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import copy
import logging
import json
import pickle

import torch.utils.data
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.miscellaneous import save_labels

from . import datasets as D
from . import samplers

from .collate_batch import BatchCollator, BBoxAugCollator
from .transforms import build_transforms


def build_dataset(dataset_list, transforms, dataset_catalog, 
                  is_train=True, dset_class=None, dset_args=None, img_caps_id=None):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_train, coco_2014_val, etc
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name)
        if dset_class is None or dset_class is "":
            dset_class = data["factory"]
        factory = getattr(D, dset_class)
        args = data["args"]
        # for COCODataset, we want to remove images without annotations
        # during training
        if "COCO" in dset_class or "VisualGenome" in dset_class:
            args["remove_images_without_annotations"] = True
        if is_train and img_caps_id != None:
            args["img_caps_id"] = img_caps_id
        if dset_class == "PascalVOCDataset":
            args["use_difficult"] = not is_train
        args["transforms"] = transforms
        args["extra_args"] = dset_args
        args["is_train"] = is_train
        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)
        dset_class = ''

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset]


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
    dataset, sampler, aspect_grouping, images_per_batch,
    num_iters=None, start_iter=0, drop_last=False
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=drop_last
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=drop_last
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_loader(cfg, is_train=True, is_distributed=False, start_iter=0, is_inference=False):
    num_gpus = get_world_size() if is_distributed else 1
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True if not cfg.DATASETS.DATASET_ARGS.DEBUG else False
        num_iters = cfg.SOLVER.MAX_ITER
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        # For inference, we want 1 img/gpu for proper mAP calculation
        # if is_inference:
            # images_per_gpu = 1
        shuffle = False if not is_distributed else True
        num_iters = cfg.TEST.MAX_ITER if cfg.TEST.MAX_ITER > 0 else None
        start_iter = 0

    if images_per_gpu > 1:
        logger = logging.getLogger(__name__)
        logger.warning(
            "When using more than one image per GPU you may encounter "
            "an out-of-memory (OOM) error if your GPU does not have "
            "sufficient memory. If this happens, you can reduce "
            "SOLVER.IMS_PER_BATCH (for training) or "
            "TEST.IMS_PER_BATCH (for inference). For training, you must "
            "also adjust the learning rate and schedule length according "
            "to the linear scaling rule. See for example: "
            "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
        )

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST

    # If bbox aug is enabled in testing, simply set transforms to None and we will apply transforms later
    transforms = None if not is_train and cfg.TEST.BBOX_AUG.ENABLED else build_transforms(cfg, is_train)
    
    # Due to curriculum learning we have multiple dataset objects one for each phase.
    if cfg.CURRICULUM.DO and is_train:
        datasets = []
        img_caps_phases_file = cfg.CURRICULUM.PHASES_FILE
        with open(img_caps_phases_file, 'rb') as f:
            img_caps_per_noun_list = pickle.load(f)

        for k,img_caps_id in enumerate(img_caps_per_noun_list):
            dataset = build_dataset(dataset_list, 
                         transforms, DatasetCatalog, is_train, 
                         cfg.DATASETS.DATASET_CLASS, cfg.DATASETS.DATASET_ARGS, img_caps_id=img_caps_id)

            datasets.append(dataset[0])
    else:
        # Could be multiple datasets if not train or multiple datasets concatenated to one if not curriculum
        datasets = build_dataset(dataset_list, 
                             transforms, DatasetCatalog, is_train, 
                             cfg.DATASETS.DATASET_CLASS, cfg.DATASETS.DATASET_ARGS)

    if is_train:
        # save category_id to label name mapping
        save_labels(datasets, cfg.OUTPUT_DIR)

    # Take care of start_iter in case of Curriculum learning
    if cfg.CURRICULUM.DO and is_train:
        curr_iters = cfg.CURRICULUM.ITERS
        cum_iter = 0
        cur_phase = 0
        for i, itter in enumerate(curr_iters):
            cum_iter += itter
            if start_iter < cum_iter:
                cum_iter = cum_iter - itter if i!= 0 else 0
                start_iter_curr = start_iter - cum_iter
                break
            cur_phase += 1

    data_loaders = []
    for i, dataset in enumerate(datasets):
        # if curriculum learning, then for phases < current phases, skip dataset;
        # for phases > current phase, start_iter = 0; only for current phase, we set the start iter
        if cfg.CURRICULUM.DO and is_train:
            if i < cur_phase:
                # Don't care
                start_iter = 0
                num_iters = cfg.CURRICULUM.ITERS[i]
            elif i == cur_phase:
                start_iter = start_iter_curr
                num_iters = cfg.CURRICULUM.ITERS[i]
            else:
                start_iter = 0
                num_iters = cfg.CURRICULUM.ITERS[i]

        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter,
            cfg.DATALOADER.DROP_LAST
        )
        collator = BBoxAugCollator() if not is_train and cfg.TEST.BBOX_AUG.ENABLED else \
            BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        data_loaders.append(data_loader)

    if is_train and not cfg.CURRICULUM.DO:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
    return data_loaders
