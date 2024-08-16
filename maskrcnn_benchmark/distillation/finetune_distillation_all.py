import argparse
import os
import datetime
import logging
import time
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
import numpy as np
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.rpn.utils import permute_and_flatten
from maskrcnn_benchmark.layers import smooth_l1_loss

def soften_proposales_iou_targets(soften_proposals, targets):
    # calculate the iou with gt and check out
    soften_proposals_indexes = []
    finetune_proposals_indexes = []
    for per_soften_proposal, per_target in zip(soften_proposals, targets):

        per_match_quality_matrix = boxlist_iou(per_target, per_soften_proposal).t()
        
        # find the index of proposals for the old classes (iou < 0.5 for all GT)
        per_iou_less_than_05_mask = per_match_quality_matrix <= 0.5
        per_iou_less_than_05_indices = torch.nonzero(per_iou_less_than_05_mask.all(dim=1)).squeeze(1)
        soften_proposals_indexes.append(per_iou_less_than_05_indices)

        # find the index of proposals for the new classes (iou > 0.5 for all GT)
        per_iou_greater_than_05_mask = per_match_quality_matrix > 0.5
        per_iou_greater_than_05_indices = torch.nonzero(per_iou_greater_than_05_mask.any(dim=1)).squeeze(1)
        finetune_proposals_indexes.append(per_iou_greater_than_05_indices)
    
    return soften_proposals_indexes, finetune_proposals_indexes


def calculate_roi_scores_distillation_losses_old_raw(soften_results, finetune_results, target_results):
    # soften: [num_proposal, 11],   finetune: [num_proposal, 11],   target: [num_proposal, 21]
    soften_scores, soften_bboxes = soften_results
    target_scores, target_bboxes = target_results
    finetune_scores, finetune_bboxes = finetune_results

    num_of_distillation_categories = soften_scores.size()[1]    #[11]
    tot_classes = target_scores.size()[1]   #[21]

    soften_labels = torch.softmax(soften_scores, dim=1)
    finetune_labels = torch.softmax(finetune_scores, dim=1)
    modified_target_scores = F.log_softmax(target_scores[:, :], dim=1)

    soften_bg_probability = soften_labels[:, 0].unsqueeze(1) 
    finetune_sums = finetune_labels.sum(dim=1, keepdim=True)
    scale_factors = soften_bg_probability / finetune_sums
    scaled_finetune_labels = finetune_labels * scale_factors

    distillation_labels = torch.cat([torch.cat([scaled_finetune_labels[:,0].unsqueeze(1), soften_labels[:,1:]],dim=1),scaled_finetune_labels[:,1:]],dim=1)
    class_distillation_loss = - distillation_labels * modified_target_scores
    class_distillation_loss_raw = torch.mean(class_distillation_loss, dim=1)  # average towards categories and proposals

    # compute distillation bbox loss
    modified_soften_boxes = soften_bboxes[:, 1:, :]  # exclude background bbox
    modified_target_bboxes = target_bboxes[:, 1:num_of_distillation_categories, :]  # exclude background bbox

    l2_loss = nn.MSELoss(size_average=False, reduce=False)
    bbox_distillation_loss = l2_loss(modified_target_bboxes, modified_soften_boxes)
    bbox_distillation_loss_raw = torch.mean(torch.sum(bbox_distillation_loss, dim=2), dim=1)  # average towards categories and proposals

    return class_distillation_loss_raw, bbox_distillation_loss_raw

def calculate_roi_scores_distillation_losses_new_raw(soften_results, finetune_results, target_results):
    # soften: [num_proposal, 11],   finetune: [num_proposal, 11],   target: [num_proposal, 21]
    soften_scores, soften_bboxes = soften_results
    target_scores, target_bboxes = target_results
    finetune_scores, finetune_bboxes = finetune_results

    num_of_distillation_categories = soften_scores.size()[1]    #[11]
    tot_classes = target_scores.size()[1]   #[21]

    soften_labels = torch.softmax(soften_scores, dim=1)
    finetune_labels = torch.softmax(finetune_scores, dim=1)
    modified_target_scores = F.log_softmax(target_scores[:, :], dim=1)

    finetune_bg_probability = finetune_labels[:, 0].unsqueeze(1)
    soften_sums = soften_labels.sum(dim=1, keepdim=True)
    scale_factors = finetune_bg_probability / soften_sums
    scaled_soften_labels = soften_labels * scale_factors

    distillation_labels = torch.cat([scaled_soften_labels,finetune_labels[:,1:]],dim=1)
    class_distillation_loss = - distillation_labels * modified_target_scores
    class_distillation_loss_raw = torch.mean(class_distillation_loss, dim=1)  # average towards categories and proposals

    # compute distillation bbox loss
    modified_finetune_boxes = finetune_bboxes[:, 1:, :]  # exclude background bbox
    modified_target_bboxes = target_bboxes[:, num_of_distillation_categories:, :]  # exclude background bbox

    l2_loss = nn.MSELoss(size_average=False, reduce=False)
    bbox_distillation_loss = l2_loss(modified_target_bboxes, modified_finetune_boxes)
    bbox_distillation_loss_raw = torch.mean(torch.sum(bbox_distillation_loss, dim=2), dim=1)  # average towards categories and proposals

    return class_distillation_loss_raw, bbox_distillation_loss_raw