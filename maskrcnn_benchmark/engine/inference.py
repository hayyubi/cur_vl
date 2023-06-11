# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import average_precision_score

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug


def compute_on_dataset(model, data_loader, device, bbox_aug, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            if bbox_aug:
                output = im_detect_bbox_aug(model, images, device)
            else:
                output = model(images.to(device))
            if timer:
                if not device.type == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        bbox_aug=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    
    # zero-shot models should have class embeddings for inference to map predicted embeddings to classes
    module = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    if hasattr(module, 'roi_heads') and 'box' in module.roi_heads:
        if module.roi_heads['box'].predictor.embedding_based:
            module.roi_heads['box'].predictor.set_class_embeddings(
                data_loader.dataset.class_emb_mtx)
    
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, bbox_aug, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)

def phased_inference(
    model,
    data_loader,
    objects_by_phases,
    device,
    do_phased_inference=False
):
    module = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    accuracy_inference = defaultdict(list)
    topk_accuracy_inference = defaultdict(list)
    Y_true = []
    Y_probs = []
    module.set_phased_inference(True)
    model.eval()

    with torch.no_grad():
        for iteration_val, (images_val, targets_val, _) in enumerate(tqdm(data_loader)):
            images_val = images_val.to(device)
            try:
                targets_val = [target.to(device) for target in targets_val]
            except:
                pass
            acc_dict, topk_acc_dict, y_true, y_probs = model(images_val, targets_val)
            for k, v in acc_dict.items():
                accuracy_inference[k].extend(v)
            for k, v in topk_acc_dict.items():
                topk_accuracy_inference[k].extend(v)
            Y_true.append(y_true)
            Y_probs.append(y_probs)

    accuracy_inference_dict = {int(k): torch.tensor(v).to(torch.float32)
                                for k, v in accuracy_inference.items()}
    topk_accuracy_inference_dict = {int(k): torch.tensor(v).to(torch.float32)
                                for k, v in topk_accuracy_inference.items()}

    Y_true = torch.cat(Y_true, dim=0).cpu().numpy()
    Y_probs = torch.cat(Y_probs, dim=0).cpu().numpy()
    cls_mAP = average_precision_score(Y_true, Y_probs, average=None)

    inference_out_dict = {}

    if do_phased_inference:
        for ph, object_ids in objects_by_phases.items():
            accuracy_list = []
            topk_accuracy_list = []
            mAP_list = []
            for object_id in object_ids:
                if object_id in accuracy_inference_dict:
                    accuracy_list.append(accuracy_inference_dict[object_id])
                if object_id in topk_accuracy_inference_dict:
                    topk_accuracy_list.append(topk_accuracy_inference_dict[object_id])
                if object_id in range(len(cls_mAP)):
                    mAP_list.append(cls_mAP[object_id])
            # Check division by 0
            if len(accuracy_list) > 0:
                inference_out_dict[f'Phase {ph} Objects Accuracy'] = torch.cat(accuracy_list, dim=0).mean().item()
            else:
                inference_out_dict[f'Phase {ph} Objects Accuracy'] = torch.tensor([-1.0])
            if len(topk_accuracy_list) > 0:
                inference_out_dict[f'Phase {ph} top-5 Objects Accuracy'] = torch.cat(topk_accuracy_list, dim=0).mean().item()
            else:
                inference_out_dict[f'Phase {ph} top-5 Objects Accuracy'] = torch.tensor([-1.0])
            if len(mAP_list) > 0:
                inference_out_dict[f'Phase {ph} Objects mean class wise mAP'] = sum(mAP_list)/len(mAP_list)
            else:
                inference_out_dict[f'Phase {ph} Objects mean class wise mAP'] = torch.tensor([-1.0])

    mean_accuracy = torch.cat(list(accuracy_inference_dict.values()), dim=0).mean().item()
    inference_out_dict[f'Mean accuracy'] = mean_accuracy
    topk_accuracy = torch.cat(list(topk_accuracy_inference_dict.values()), dim=0).mean().item()
    inference_out_dict[f'top-5 Mean accuracy'] = topk_accuracy
    mean_class_wise_mAP = sum(cls_mAP[1:])/(len(cls_mAP)-1)
    inference_out_dict[f'Mean Class wise mAP'] = mean_class_wise_mAP

    return inference_out_dict