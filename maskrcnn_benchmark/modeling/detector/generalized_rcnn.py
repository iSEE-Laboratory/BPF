# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
import torch.nn.functional as F
from torch import nn
import random
import numpy as np
import pdb
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.layers import smooth_l1_loss

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..rpn.utils import permute_and_flatten
from maskrcnn_benchmark.modeling.attention_map import generate_attention_map

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):

        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        # ADDED by A. Geraci to use UCE (as in Modeling the Background)
        self.incremental = cfg.INCREMENTAL
        self.n_old_cl = len(cfg.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES)
        self.n_new_cl = len(cfg.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES)
        self.finetune_enable = cfg.FINETUNE.ENABLE
        self.cfg = cfg
        if not cfg.MODEL.RPN.EXTERNAL_PROPOSAL:
            print('generalized_rcnn.py | Do not use external proposals, so use RPN.')
            self.rpn = build_rpn(cfg, self.backbone.out_channels)
        else:
            print('generalized_rcnn.py | Use external proposals.')

        # here, adding we use cfg.INCREMENTAL to use unbiased CE loss as in MiB
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None, pseudo_targets = None, rpn_output_source=None, features=None, proposals=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if features is not None and proposals is not None:
            target_scores, target_bboxes, mask_logits, roi_align_features = self.roi_heads.calculate_soften_label(features, proposals)
            return (target_scores, target_bboxes), mask_logits, roi_align_features
        else:
            images = to_image_list(images)

            features, backbone_features = self.backbone(images.tensors)

            if self.cfg.UNK.ENABLE:
                attention_maps = generate_attention_map(features=features)
                (proposals, proposal_losses), anchors, rpn_output = self.rpn(images, features, targets, rpn_output_source, attention_maps=attention_maps)
            else:
                (proposals, proposal_losses), anchors, rpn_output = self.rpn(images, features, targets, rpn_output_source)

            if self.roi_heads:
                if self.training:
                    #x, result, soften_results, detector_losses, roi_align_features = self.roi_heads(features, proposals, targets, pseudo_targets,iteration)
                    if self.cfg.UNK.ENABLE:
                        result, detector_losses = self.roi_heads(features, proposals, targets, pseudo_targets, attention_maps=attention_maps)
                    else:
                        result, detector_losses = self.roi_heads(features, proposals, targets, pseudo_targets)
                else:
                    x, result, results_background, _ = self.roi_heads(features, targets)
                    return result, features, results_background             
                proposals = result
            else:
                # RPN-only models don't have roi_heads
                x = features
                result = proposals
                detector_losses = {}

            if self.training:
                losses = {}
                losses.update(detector_losses)
                losses.update(proposal_losses)
                #return losses, features, backbone_features, anchors, rpn_output, proposals, roi_align_features, soften_results
                return losses, features, rpn_output

            return result, features,

    def use_external_proposals_edgeboxes(self, images, proposals, targets=None):

        if self.training and targets is None:
            raise ValueError("In external proposal training mode, targets should be passed")
        if proposals is None:
            raise ValueError("In external proposal mode, proposals should be passed")

        images = to_image_list(images)
        features, backbone_features = self.backbone(images.tensors)

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:  # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            return losses

        return result

    def new_dataset_finetune_old_model(self, images, proposals, targets=None):

        if self.training and targets is None:
            raise ValueError("In external proposal training mode, targets should be passed")
        if proposals is None:
            raise ValueError("In external proposal mode, proposals should be passed")

        images = to_image_list(images)
        features, backbone_features = self.backbone(images.tensors)


        class_logits = self.rpn.feature_extraction(features)

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:  # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            return losses

        return result

    def generate_soften_proposal(self, images, targets=None):

        images = to_image_list(images)  # convert images to image_list type
        features, backbone_features = self.backbone(images.tensors)  # extra image features from backbone network
        (all_proposals, proposal_losses), anchors, rpn_output = self.rpn(images, features, targets)  # use RPN to generate ROIs

        all_selected_proposals = []
        for k in range(len(all_proposals)):
            # sort proposals according to their objectness score
            inds = [all_proposals[k].get_field("objectness").sort(descending=True)[1]]
            proposals = all_proposals[k][inds]
            num_proposals = len(proposals)

            # get proposal information: bbox, objectness score, proposal mode & image size
            proposal_bbox = proposals.bbox
            proposal_score = proposals.get_field("objectness")
            proposal_mode = proposals.mode
            image_size = proposals.size

            # choose first 128 highest objectness score proposals  and then random choose 64 proposals from them
            if num_proposals < 64:
                list = range(0, num_proposals, 1)
                selected_proposal_index = random.sample(list, num_proposals)
            elif num_proposals < 128:
                list = range(0, num_proposals, 1)
                selected_proposal_index = random.sample(list, 64)
            else:
                list = range(0, 128, 1)
                selected_proposal_index = random.sample(list, 64)

            for i, element in enumerate(selected_proposal_index):
                if i == 0:
                    selected_proposal_bbox = proposal_bbox[element]
                    selected_proposal_bbox = selected_proposal_bbox.view(-1, 4)
                    selected_proposal_score = proposal_score[element].view(-1, 1)
                else:
                    selected_proposal_bbox = torch.cat((selected_proposal_bbox, proposal_bbox[element].view(-1, 4)), 0)  # vertical tensor cascated
                    selected_proposal_score = torch.cat((selected_proposal_score, proposal_score[element].view(-1, 1)), 1)  # horizontal cascate tensors
            selected_proposal_bbox = selected_proposal_bbox.view(-1, 4)
            selected_proposal_score = selected_proposal_score.view(-1)
            selected_proposals = BoxList(selected_proposal_bbox, image_size, proposal_mode)
            selected_proposals.add_field("objectness", selected_proposal_score)
            all_selected_proposals.append(selected_proposals)
        # generate soften proposal labels
        soften_scores, soften_bboxes, mask_logits, roi_align_features = self.roi_heads.calculate_soften_label(features, all_selected_proposals)  # use ROI-subnet to generate final results

        return (soften_scores, soften_bboxes), mask_logits, all_selected_proposals, features, backbone_features, anchors, rpn_output, roi_align_features

    def generate_soften_label_external_proposal(self, images, proposals, targets=None):

        images = to_image_list(images)  # convert images to image_list type
        features, backbone_features = self.backbone(images.tensors)  # extra image features from backbone network

        # get proposal information: bbox, proposal mode & image size
        proposal_bbox = proposals[0].bbox
        proposal_mode = proposals[0].mode
        image_size = proposals[0].size

        # choose first 128 highest objectness score proposals  and then random choose 64 proposals from them
        list = range(0, 128, 1)
        selected_proposal_index = random.sample(list, 64)
        for i, element in enumerate(selected_proposal_index):
            if i == 0:
                selected_proposal_bbox = proposal_bbox[element]
                selected_proposal_bbox = selected_proposal_bbox.view(-1, 4)
            else:
                selected_proposal_bbox = torch.cat((selected_proposal_bbox, proposal_bbox[element].view(-1, 4)), 0)  # vertical tensor cascated
        selected_proposal_bbox = selected_proposal_bbox.view(-1, 4)
        selected_proposals = BoxList(selected_proposal_bbox, image_size, proposal_mode)
        selected_proposals = [selected_proposals]

        # generate soften labels
        soften_scores, soften_bboxes = self.roi_heads.calculate_soften_label(features, selected_proposals, targets)

        return (soften_scores, soften_bboxes), selected_proposals


    def feature_extraction_by_rpn(self, features):
        class_logits = self.rpn.feature_extraction(features)
        return class_logits

    def generate_pseudo_targets(self, images):
        #pdb.set_trace()
        images = to_image_list(images)

        features, _ = self.backbone(images.tensors)

        (proposals, _), _, _ = self.rpn(images, features)

        x, result, results_background, _ = self.roi_heads.get_pseudo_labels(features, proposals)

        return result, features, results_background

    def generate_features_rpn_output(self, images):

        images = to_image_list(images)  # convert images to image_list type
        features, backbone_features = self.backbone(images.tensors)  # extra image features from backbone network
        (all_proposals, proposal_losses), anchors, rpn_output = self.rpn(images, features, None)  # use RPN to generate ROIs

        return features, rpn_output
