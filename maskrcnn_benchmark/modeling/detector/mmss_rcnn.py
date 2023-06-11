"""
Implements the Multimedia Self-Supervised Grid-based (proposal-free) CNN framework
"""
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from collections import defaultdict

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from maskrcnn_benchmark.config import get_cfg_defaults
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

from ..backbone import build_backbone
from ..language_backbone import build_language_backbone
from ..mmss_heads import build_mmss_heads
from .generalized_rcnn import GeneralizedRCNN
from sklearn.metrics import average_precision_score


class MMSSRegionModel(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg, spatial_dropout=100):
        # Whether to use pretrained weight for language backbone
        super(MMSSRegionModel, self).__init__()
        if not cfg.MODEL.LANGUAGE_BACKBONE.USE_PRETRAINED:
            assert cfg.MODEL.LANGUAGE_BACKBONE.FREEZE == False

        # Whether to use fixed embeddings for computing class scores
        self.pretrained_class_emb = cfg.MODEL.LANGUAGE_BACKBONE.FREEZE

        # __forward__() can't be used until these are initialized, AFTER the optimizer is made.
        self.emb_dim = None
        self.num_classes = None
        self.cls_score = None

        # Set appropriate way of getting bounding box if not using ground truth ones for training.
        self.bbox_type = cfg.MODEL.BBOX.TYPE
        if self.bbox_type == "RP":
            rpn_cfg = get_cfg_defaults()
            rpn_cfg.merge_from_file(cfg.MODEL.BBOX.RPN.CFG)
            rpn_cfg.freeze()
            self.rpn = GeneralizedRCNN(rpn_cfg)
            checkpointer = DetectronCheckpointer(
                rpn_cfg, self.rpn, save_dir=cfg.MODEL.BBOX.RPN.WEIGHT, load_classifier=False)
            checkpointer.load(load_trainer_state=False)
            for p in self.rpn.parameters():
                p.requires_grad = False

        self.backbone = build_backbone(cfg)
        self.bbox_feature_extractor = make_roi_box_feature_extractor(cfg, self.backbone.out_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.language_backbone = build_language_backbone(cfg) 
 
        self.mmss_heads = build_mmss_heads(cfg,
            v_dim=self.bbox_feature_extractor.out_channels,
            l_dim=self.language_backbone.out_channels,
            loc_dim=5,
            backbone=self.language_backbone.body,
        )
        self.mvm = cfg.MODEL.MMSS_HEAD.TRANSFORMER.MASKED_VISUAL_MODELING
        cfg_spatial_dropout = cfg.MODEL.MMSS_HEAD.SPATIAL_DROPOUT
        self.spatial_dropout =  cfg_spatial_dropout if cfg_spatial_dropout > 0 else spatial_dropout
        self.scale_TH = cfg.MODEL.MMSS_HEAD.SCALE_TH

        self.inference = cfg.INFERENCE.DO
        self.phased_inference = False
        self.inference_from_image = False
        self.class_list = None
        self.need_cls_embedding_reset = 0
        self.prev_knowledge_source_mode = False
        
    def forward(self, images, targets, iter_percent=None, prev_knowledge=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[str]): ground-truth captions for images (optional)

        Returns:
            result tuple: (dict[Tensor], dict[Tensor]): losses and other information.

        """
        images = to_image_list(images)
        batch_sz, _, image_h, image_w = images.tensors.shape

        # If not using pretrained class embedding reset it on the first iteration while
        # doing validation/inferencing
        if not self.pretrained_class_emb:
            if self.training:
                self.need_cls_embedding_reset = 1
            else:
                self.reset_class_embeddings()

        # During inference, only images may be passed as well from dataset like ImageNet
        if (self.inference or self.phased_inference) and not isinstance(targets[0], tuple):
            det_targets = targets
        else:
            captions, det_targets = zip(*targets)
            captions, det_targets = list(captions), list(det_targets)
        det_targets = [det_target.to(images.tensors.device) for det_target in det_targets]

        visual_grid_features = self.backbone(images.tensors)

        # If inferencing get accuracy results and return
        if self.inference:
            result = self.do_inference(visual_grid_features, det_targets)
            return result
        elif self.phased_inference:
            result = self.do_phased_inference(visual_grid_features, det_targets)
            return result

        # If bbox type is from ground truth or a rpn network
        if self.bbox_type == "GT":
            BBox = det_targets
        elif self.bbox_type == "RP":
            self.rpn.eval()
            BBox = self.rpn(images)
        else:
            raise NotImplementedError("Automatic BBox from GT or RP only")
        bbox_features = self.bbox_feature_extractor(visual_grid_features, BBox)
        bbox_features = self.avgpool(bbox_features)

        tot_bbox, _, _, _ = bbox_features.shape
        bbox_features = bbox_features.reshape(tot_bbox, -1)
        bbox_features_list = []
        bbox_loc_list = []
        num_bbox_list = []
        start_iter = 0
        for bbox_list in BBox:
            # Keep count of num of bbox in each image to make mask
            num_bbox = len(bbox_list)
            num_bbox_list.append(num_bbox)
            # Normalize "xyxy" of bbox for loc
            bbox = bbox_list.bbox.clone()
            bbox[:, [0,2]] /= image_w 
            bbox[:, [1,3]] /= image_h 
            area = bbox_list.area() / (image_h * image_w)
            bbox_loc_list.append(torch.cat((bbox, area[:, None]), dim=1))
            # Per image features
            end_iter = start_iter + num_bbox
            bbox_features_list.append(bbox_features[start_iter:end_iter,:])
            start_iter = end_iter

        flattened_features = torch.nn.utils.rnn.pad_sequence(
            bbox_features_list, batch_first=True)
        flattened_loc = torch.nn.utils.rnn.pad_sequence(
            bbox_loc_list, batch_first=True)
        flattened_mask = np.zeros([batch_sz, flattened_features.shape[1]], dtype=np.uint8)
        for i, num_bbox in enumerate(num_bbox_list):
            flattened_mask[i, :num_bbox] = 1

        num_regions = min(flattened_features.shape[1], self.spatial_dropout)
        flattened_features = flattened_features[:, :num_regions, :]
        flattened_loc = flattened_loc[:, :num_regions, :]
        flattened_mask = flattened_mask[:, :num_regions]

        input_image = {
            'region_features': flattened_features,
            'region_mask': torch.tensor(flattened_mask).cuda(),
            'region_loc': flattened_loc,
            'mvm_mask': torch.zeros(batch_sz, num_regions).cuda(),
            'target_region_features': flattened_features,
        }
        if self.mvm:
            raise NotImplementedError

        input_caption = self.language_backbone(captions)

        if self.prev_knowledge_source_mode:
            assert 'GroundingHead' in self.mmss_heads
            return self.mmss_heads['GroundingHead'](input_image, input_caption)

        mmss_outputs = {}
        mmss_losses = {}
        for head in self.mmss_heads:
            o, l = self.mmss_heads[head](input_image, input_caption, 
                                        iter_percent=iter_percent, images=images, 
                                        targets=targets, prev_knowledge=prev_knowledge)
            if self.scale_TH >= 0:
                if head == "TransformerHead":
                    for loss_type in l:
                        l[loss_type] *= self.scale_TH
                elif head == "GroundingHead":
                    for loss_type in l:
                        l[loss_type] *= (1 - self.scale_TH)
                else:
                    raise NotImplementedError("Loss scaling only implemented for \
                        Transformer Head and Grounding Head")
            mmss_outputs.update(o)
            mmss_losses.update(l)

        for v in mmss_losses.values():
            if torch.isnan(v):
                print(self.mmss_heads['GroundingHead'].log_info)
                warnings.warn("Nan in mmss head", RuntimeWarning)

        # During training, det_targets contain many classes which we do not test for
        if not self.training:
            with torch.no_grad():
                acc_dict = self.do_inference(visual_grid_features, det_targets)
            mmss_outputs.update(acc_dict)

        return mmss_outputs, mmss_losses

    def set_prev_knowledge_source_mode(self):
        self.prev_knowledge_source_mode = True
        assert 'GroundingHead' in self.mmss_heads
        self.mmss_heads['GroundingHead'].prev_knowledge_source_mode = True

    def set_prev_knowledge_model(self, model):
        assert 'GroundingHead' in self.mmss_heads
        self.mmss_heads['GroundingHead'].prev_knowledge_model = model

    def check_if_prev_knowledge_model_is_set(self):
        return self.mmss_heads['GroundingHead'].prev_knowledge_model is not None

    def set_class_list(self, class_list):
        self.class_list = class_list

    def set_class_embeddings(self, embs=None):
        device = self.mmss_heads['GroundingHead'].v2l_projection.weight.device
        # Set cls_score skeleton at the begining of training and embedding when
        # running the model with pretrained embeddings
        self.emb_dim = self.language_backbone.out_channels
        self.num_classes = embs.shape[0] if embs is not None else len(self.class_list)
        self.cls_score = nn.Linear(self.emb_dim, self.num_classes)
        self.cls_score.to(device)

        if self.pretrained_class_emb:
            assert embs is not None
            self.cls_score.weight.data = torch.tensor(embs, device=device, 
                                                      requires_grad=False)
            self.cls_score.bias.data = torch.zeros_like(self.cls_score.bias.data, 
                                                    requires_grad=False)

    def reset_class_embeddings(self):
        # Reset class embeddings only the first need_cls_embedding_reset bit is flipped from 1 to 0
        if self.need_cls_embedding_reset == 0:
            return
        # Enter here only if using unfrozen language model
        if not self.pretrained_class_emb:
            # Otherwise some classes will be masked resulting in Nan cls_score
            orig_mlm_value = self.language_backbone[0].mlm
            self.language_backbone[0].mlm = False

            device = self.mmss_heads['GroundingHead'].v2l_projection.weight.device
            encoded_class_list = self.language_backbone(self.class_list)
            mask = (1 - encoded_class_list['special_tokens_mask']).to(torch.float32)
            embs = (encoded_class_list['input_embeddings'] * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
            self.cls_score.weight.data = embs.clone().detach()
            self.cls_score.bias.data = torch.zeros_like(self.cls_score.bias.data)
            self.need_cls_embedding_reset = 0

            # Reset language backbone value
            self.language_backbone[0].mlm = orig_mlm_value

    def set_inference_from_image_true(self):
        self.inference_from_image = True

    def set_phased_inference(self, val):
        self.phased_inference = val

    def do_inference(self, in_features, targets):
        probs, target_classes = self.get_class_probs_and_target_classes(in_features, targets)

        # Calculate Class wise Accuracy and mAP
        correctness = torch.argmax(probs, dim=-1) == target_classes

        # Top K accuracy
        k=5
        _, topk_class = torch.topk(probs, 5)
        topk_equality = torch.eq(topk_class, target_classes[:, None].expand_as(topk_class))
        topk_correctness = torch.sum(topk_equality, dim=-1)

        class_accuracy_dict = defaultdict(list)
        y_true = []
        y_pred = []
        seen_class = set()
        for i, cl in enumerate(target_classes.tolist()):
            class_accuracy_dict[str(cl)].append(correctness[i])
            if cl not in seen_class:
                cl_true = (target_classes == cl)
                cl_probs = probs[:, cl]
                y_true.append(cl_true)
                y_pred.append(cl_probs)
                seen_class.add(cl)

        class_accuracy_list = [torch.tensor(v).to(torch.float32).mean()
                                    for k, v in class_accuracy_dict.items()]
        class_wise_accuracy = (sum(class_accuracy_list) / len(class_accuracy_list)).cuda()

        y_true = torch.stack(y_true, dim=1).cpu().numpy()
        y_pred = torch.stack(y_pred, dim=1).cpu().numpy()
        mAP = torch.tensor(average_precision_score(y_true, y_pred)).cuda()

        result_dict = {"Zeroshot Recognition mean class accuracy": class_wise_accuracy,
                        "Zeroshot Recognition mAP": mAP,
                        "Zeroshot Recognition Accuracy": correctness.to(torch.float32).mean(),
                        "Top-5 Zeroshot Recognition Accuracy": topk_correctness.to(torch.float32).mean()}

        return result_dict

    def do_phased_inference(self, in_features, targets):
        probs, target_classes = self.get_class_probs_and_target_classes(in_features, targets)
        correctness = torch.argmax(probs, dim=-1) == target_classes

        k=5
        _, topk_class = torch.topk(probs, 5)
        topk_equality = torch.eq(topk_class, target_classes[:, None].expand_as(topk_class))
        topk_correctness = torch.sum(topk_equality, dim=-1)

        class_accuracy_dict = defaultdict(list)
        topk_class_accuracy_dict = defaultdict(list)
        for i, cl in enumerate(target_classes.tolist()):
            class_accuracy_dict[str(cl)].append(correctness[i])
            topk_class_accuracy_dict[str(cl)].append(topk_correctness[i])

        y_labels = target_classes
        y_probs = probs
        y_true = torch.zeros(y_probs.shape).cuda()
        y_true[torch.arange(y_true.shape[0]).cuda(), y_labels] = 1

        return class_accuracy_dict, topk_class_accuracy_dict, y_true, y_probs
    
    def get_class_probs_and_target_classes(self, in_features, targets):
        if not self.inference_from_image:
           BBox = targets
           bbox_features = self.bbox_feature_extractor(in_features, BBox)
        else:
            # TODO: Implement extracting features from ResNet Head 
            # rather than using pooling then head for whole images
            # It is assumed for now that while working with whole images,
            # the class labels will also be in the FasterRCNN BoxList format.
            raise NotImplementedError("Extracting features for whole image not yet implemented")
        bbox_features = self.avgpool(bbox_features)
        tot_bbox, _, _, _ = bbox_features.shape
        bbox_features = bbox_features.reshape(tot_bbox, -1)

        cls_emb = self.mmss_heads['GroundingHead'].v2l_projection(bbox_features)
        cls_logit = self.cls_score(cls_emb)
        probs = F.softmax(cls_logit, -1)
        target_classes = [target.get_field("labels") for target in targets]
        target_classes = torch.cat(target_classes).to(torch.int64)

        return probs, target_classes