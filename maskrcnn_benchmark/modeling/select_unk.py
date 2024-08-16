import torch
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou


def select_unk_idxes(proposals, targets, obj_topk=0.01, attn_topk=0.01, unk_iou_gt=0.1):
    unk_idxes = []
    for img_idx in range(len(proposals)):
        proposals_per_image = proposals[img_idx]
        targets_per_image = targets[img_idx]
        match_quality_matrix = boxlist_iou(targets_per_image, proposals_per_image)

        per_objectness = proposals_per_image.get_field("objectness")
        per_attention = proposals_per_image.get_field("attention")

        obj_topk_num = int(len(proposals_per_image)* obj_topk)
        attn_topk_num = int(len(proposals_per_image)* attn_topk)
        unk_topk_objectness_idx = torch.topk(per_objectness, obj_topk_num, largest=True)[1]
        unk_topk_attention_idx = torch.topk(per_attention, attn_topk_num, largest=True)[1]
        unk_indices = set(unk_topk_objectness_idx.tolist()).intersection(unk_topk_attention_idx.tolist())
        unk_indices = torch.tensor(list(unk_indices))
        # print("unk_indices size: ",unk_indices.size())
        if len(unk_indices) > 0:
            # match_quality_matrix_t: [num_proposals, num_targets]
            match_quality_matrix_t = match_quality_matrix.t()
            # unk_matrix: [num_unk, num_targets]
            unk_matrix = match_quality_matrix_t[unk_indices]    
            unk_iou_mask = unk_matrix < unk_iou_gt
            unk_iou_thres_mask = unk_iou_mask.all(dim=1)
            unk_iou_indices = torch.nonzero(unk_iou_thres_mask).squeeze(1)
            if len(unk_iou_indices) > 0:
                unk_idx = unk_indices[unk_iou_indices]
                unk_idxes.append(unk_idx) 
            else:
                unk_idxes.append([])
        else:
            unk_idxes.append([])
        
    return unk_idxes
