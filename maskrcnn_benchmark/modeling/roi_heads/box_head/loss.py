# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import torch
from torch import nn
from torch.nn import functional as F
import os
import pdb
from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat
import maskrcnn_benchmark.utils.comm as comm
from maskrcnn_benchmark.utils.store import Store
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.draw import Draw_singe_image
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import boxlist_nms
from maskrcnn_benchmark.modeling.attention_map import calculate_attention_scores_per_img_roi_align


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
        self, 
        proposal_matcher, 
        fg_bg_sampler, 
        box_coder, 
        cls_agnostic_bbox_reg=False,
        uce=False,
        old_classes=[],
        cfg=None
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.uce = uce
        self.n_old_cl = len(old_classes)
        self.n_cl = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES # num_fg + 1

        # UNK PARAMETERS
        self.unk_enable = cfg.UNK.ENABLE
        self.obj_topk = cfg.UNK.OBJ_TOPK
        self.attn_topk = cfg.UNK.ATTN_TOPK
        self.unk_iou_gt = cfg.UNK.IOU_GT

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        #target = target.copy_with_fields("labels")
        target = target.copy_with_fields(["labels","weights"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets, match_quality_matrix

    def prepare_targets(self, proposals, targets, attention_maps=None):
        labels = []
        regression_targets = []
        weights = []
        match_proposals_targets = []
        if self.unk_enable:
            unk_idxes = []
            unk_proposals = []
        # for proposals_per_image, targets_per_image in zip(proposals, targets):
        for img_idx in range(len(proposals)):
            proposals_per_image = proposals[img_idx]
            targets_per_image = targets[img_idx]

            matched_targets, match_quality_matrix = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # unknow object
            if self.unk_enable:
                bg_idx = bg_inds.nonzero()
                bg_anchors = proposals_per_image[bg_idx.view(-1)]
                neg_attention = calculate_attention_scores_per_img_roi_align(bg_anchors, attention_maps[img_idx])
                neg_objectness = proposals_per_image.get_field("objectness")[bg_idx]

                obj_topk_num = int(len(bg_idx)* self.obj_topk)
                attn_topk_num = int(len(bg_idx)* self.attn_topk)
                unk_topk_objectness_idx = torch.topk(neg_objectness.squeeze(1), obj_topk_num, largest=True)[1]
                unk_topk_attention_idx = torch.topk(neg_attention.squeeze(1), attn_topk_num, largest=True)[1]
                unk_indices = set(unk_topk_objectness_idx.tolist()).intersection(unk_topk_attention_idx.tolist())
                unk_indices = torch.tensor(list(unk_indices))
                # print("unk_indices size: ",unk_indices.size())
                unk_proposals_per_img = BoxList(torch.empty(0, 4), proposals_per_image.size, mode="xyxy")
                if len(unk_indices) > 0:
                    unk_idx = bg_idx[unk_indices].squeeze(1)    # [num_unk]
                    # match_quality_matrix_t: [num_proposals, num_targets]
                    match_quality_matrix_t = match_quality_matrix.t()
                    # unk_matrix: [num_unk, num_targets]
                    unk_matrix = match_quality_matrix_t[unk_idx]    
                    unk_iou_mask = unk_matrix < self.unk_iou_gt
                    unk_iou_thres_mask = unk_iou_mask.all(dim=1)
                    unk_iou_indices = torch.nonzero(unk_iou_thres_mask).squeeze(1)
                    if len(unk_iou_indices) > 0:
                        # unk_idx: the index of unknown bbox in initial proposal indexs
                        unk_idx = unk_idx[unk_iou_indices]
                        labels_per_image[unk_idx] = -1
                        unk_proposals_per_img = proposals_per_image[unk_idx.view(-1)]            
                
                # add unk proposals
                unk_proposals.append(unk_proposals_per_img)


            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # add bg proposals
            # bg_idx = (labels_per_image == 0).nonzero()
            # bg_proposals_per_img = proposals_per_image[bg_idx.view(-1)]
            # bg_proposals.append(bg_proposals_per_img)

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )

            weight_target_per_image = matched_targets.get_field("weights")

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
            weights.append(weight_target_per_image)
            match_proposals_targets.append(match_quality_matrix)

        if self.unk_enable:
            return labels, regression_targets, weights, unk_idxes, unk_proposals
        else:
            return labels, regression_targets, weights

    def subsample(self, proposals, targets, attention_maps=None):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """
        if self.unk_enable:
            labels, regression_targets, weights, unk_idxes, unk_proposals = self.prepare_targets(proposals, targets, attention_maps=attention_maps)
        else:
            labels, regression_targets, weights = self.prepare_targets(proposals, targets)

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, regression_targets_per_image, weights_targets_per_image, proposals_per_image in zip(
            labels, regression_targets, weights, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )
            proposals_per_image.add_field("weights", weights_targets_per_image)

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals

        return proposals


    def __call__(self, class_logits, box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """
        # print('box_head | loss.py | class_logits size {0}'.format(class_logits[0].size()))
        # print('box_head | loss.py | box_regression size {0}'.format(box_regression[0].size()))
        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )

        if self.uce:
            #objectnesses = cat([proposal.get_field("objectness") for proposal in proposals], dim=0)
            #high_obj_proposals = (objectnesses > 0.5).nonzero(as_tuple=True)[0]

            outputs = torch.zeros_like(class_logits)
            den = torch.logsumexp(class_logits, dim=1)  # B, H, W       den of softmax
            outputs[:, 0] = torch.logsumexp(class_logits[:, 0:self.n_old_cl+1], dim=1) - den  # B, H, W       p(O)
            outputs[:, self.n_old_cl+1:] = class_logits[:, self.n_old_cl+1:] - den.unsqueeze(dim=1)  # B, N, H, W    p(N_i)

            labels = labels.clone()  # B, H, W

            classification_loss = F.nll_loss(outputs, labels)

        else:
            sample_weights = cat([proposal.get_field("weights") for proposal in proposals], dim=0)
            per_sample_loss = F.cross_entropy(class_logits, labels, reduction='none')
            weighted_loss = per_sample_loss * sample_weights
            classification_loss = weighted_loss.mean()            
            #classification_loss = F.cross_entropy(class_logits, labels)

        # get indices that correspond to the regression targets for the corresponding ground truth labels, to be used
        # with advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3], device=device)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            size_average=False,  # sum
            beta=1,
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = FastRCNNLossComputation(
        matcher, 
        fg_bg_sampler, 
        box_coder, 
        cls_agnostic_bbox_reg,
        cfg.INCREMENTAL,
        cfg.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES,
        cfg
    )

    return loss_evaluator
