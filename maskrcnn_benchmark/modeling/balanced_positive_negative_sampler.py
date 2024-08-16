# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import sys
import pdb
from maskrcnn_benchmark.modeling.draw import draw_singe_image, Draw_singe_image
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.layers import nms as _box_nms

class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentace of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs, objectness=None, proposals=None, targets=None, attention_maps=None, match_matrix=None, img_id=None, unk_num=None, images=None):
        """
        Arguments:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """

        pos_idx = []
        neg_idx = []
        
        if proposals is None:
            for matched_idxs_per_image in matched_idxs:
                positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
                negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)

                num_pos = int(self.batch_size_per_image * self.positive_fraction)
                # protect against not enough positive examples
                num_pos = min(positive.numel(), num_pos)
                num_neg = self.batch_size_per_image - num_pos
                # protect against not enough negative examples
                num_neg = min(negative.numel(), num_neg)

                # randomly select positive and negative examples
                perm_pos = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
                perm_neg = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

                pos_idx_per_image = positive[perm_pos]
                neg_idx_per_image = negative[perm_neg]

                # create binary mask from indices
                pos_idx_per_image_mask = torch.zeros_like(
                    matched_idxs_per_image, dtype=torch.uint8
                )
                neg_idx_per_image_mask = torch.zeros_like(
                    matched_idxs_per_image, dtype=torch.uint8
                )
                pos_idx_per_image_mask[pos_idx_per_image] = 1
                neg_idx_per_image_mask[neg_idx_per_image] = 1

                pos_idx.append(pos_idx_per_image_mask)
                neg_idx.append(neg_idx_per_image_mask)

        # else:
        #     num_unk = unk_num
        #     unk_thresh = 0.1
        #     unk_obj_thresh = 0.75
        #     nms_thresh = 0.5
        #     for matched_idxs_per_image, proposal_per_image, per_attention_map, target_per_image, match_matrix_per_image, id_per_img, per_img in zip(matched_idxs, proposals, targets, match_matrix, img_id, images):
                
        #         positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
        #         negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1) # 属于负样本的index

        #         num_pos = int(self.batch_size_per_image * self.positive_fraction)
        #         # protect against not enough positive examples
        #         num_pos = min(positive.numel(), num_pos)
        #         num_neg = self.batch_size_per_image - num_pos
        #         # protect against not enough negative examples
        #         num_neg = min(negative.numel(), num_neg)

        #         # randomly select positive and negative examples
        #         perm_pos = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
                
        #         neg_all = torch.randperm(negative.numel(), device=negative.device)
        #         perm_neg = neg_all[:num_neg]
        #         extra_neg = neg_all[num_neg:]

        #         pos_idx_per_image = positive[perm_pos]
        #         neg_idx_per_image = negative[perm_neg]  # 表示的是，采样的负样本的下标

        #         objectness = proposal_per_image.get_field("objectness")
        #         objectness_neg = objectness[neg_idx_per_image]  #已采样的负样本的分数
        #         attention = proposal_per_image.get_field("attention")
        #         attention_neg = attention[neg_idx_per_image]

        #         # 对于每个负样本proposal，检查是否所有与targets的IOU都小于0.1
        #         low_iou_mask = (match_matrix_per_image[:,neg_idx_per_image] < unk_thresh).all(dim=0)
        #         not_low_iou_mask = ~low_iou_mask
        #         objectness_neg[not_low_iou_mask] = -1.0

        #         # 选出前num_unk个objectness分数最高且大于0.75的索引
        #         _, top_indices = objectness_neg.topk(num_unk, largest=True)
        #         high_objectness_mask = objectness_neg[top_indices] > unk_obj_thresh
        #         del_indices = top_indices[high_objectness_mask]

        #         removed_neg_idx = neg_idx_per_image[del_indices]
        #         removed_bbox = proposal_per_image.bbox[removed_neg_idx]
        #         removed_bbox_scores = objectness[removed_neg_idx]

        #         keep = _box_nms(removed_bbox, removed_bbox_scores, nms_thresh)
        #         del_indices = del_indices[keep]

        #         # 使用布尔索引移除样本
        #         mask = torch.ones_like(neg_idx_per_image, dtype=torch.bool)
        #         mask[del_indices] = False
        #         neg_idx_per_image = neg_idx_per_image[mask]

        #         # 画图
        #         # added_bbox = proposal_per_image.bbox[extra_neg_idx]
        #         # added_proposal_per_image = BoxList(added_bbox, proposal_per_image.size)

        #         # if len(removed_proposal_per_image) > 0:
        #             # removed_proposal_per_image = BoxList(removed_bbox, proposal_per_image.size)
        #             # removed_proposal_per_image.add_field("scores",removed_bbox_scores)
        #             # removed_proposal_per_image = boxlist_nms(removed_proposal_per_image)
        #             # Draw_singe_image(id_per_img, per_img, target_per_image, removed_proposal_per_image)

        #         # create binary mask from indices
        #         pos_idx_per_image_mask = torch.zeros_like(
        #             matched_idxs_per_image, dtype=torch.uint8
        #         )
        #         neg_idx_per_image_mask = torch.zeros_like(
        #             matched_idxs_per_image, dtype=torch.uint8
        #         )
        #         pos_idx_per_image_mask[pos_idx_per_image] = 1
        #         neg_idx_per_image_mask[neg_idx_per_image] = 1

        #         pos_idx.append(pos_idx_per_image_mask)
        #         neg_idx.append(neg_idx_per_image_mask)            
        # return pos_idx, neg_idx
     
        return pos_idx, neg_idx


def boxlist_nms(boxlist, nms_thresh=0.5, max_proposals=-1, score_field="scores"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    #pdb.set_trace()
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)



# 从未被采样的负样本中补充前景分数最小的num_unk个样本
# remaining_negatives = negative[extra_neg]
# remaining_scores = objectness[remaining_negatives]
# _, lowest_scores_indices = remaining_scores.topk(num_unk, largest=False)
# # 从剩余样本中选取
# extra_neg_idx = remaining_negatives[lowest_scores_indices]
# neg_idx_per_image = torch.cat((neg_idx_per_image, extra_neg_idx), dim=0)