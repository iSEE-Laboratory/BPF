import torch
import pdb
from maskrcnn_benchmark.structures.bounding_box import BoxList


def filter_pseudo_boxes_with_iou(pseudo_targets, gt_targets, low_thres=0.4, high_thres=0.7):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes of (pseudo_targets), sized [N,4].
      box2: (tensor) bounding boxes of (gt_targets), sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    #pdb.set_trace()
    all_target = []
    low_iou_target = []
    high_iou_target = []
    #pseudo_boxes = [x.bbox for x in pseudo_targets] #[N, num_pse, 4]
    #gt_boxes = [x.bbox for x in gt_targets] #[N, num_gt, 4]

    for per_pse, per_gt in zip(pseudo_targets, gt_targets):
        # 首先计算两个box左上角点坐标的最大值和右下角坐标的最小值，然后计算交集面积，最后把交集面积除以对应的并集面积
        box1 = per_pse.bbox
        box2 = per_gt.bbox
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(  # 左上角的点
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(  # 右下角的点
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 指两个box没有重叠区域
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2]-box1[:, 0]) * (box1[:, 3]-box1[:, 1])  # [N,]
        area2 = (box2[:, 2]-box2[:, 0]) * (box2[:, 3]-box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        
        low_iou_row_mask = iou <= low_thres
        low_iou_valid_rows = low_iou_row_mask.all(dim=1)
        low_iouvalid_row_indices = torch.nonzero(low_iou_valid_rows).squeeze()
        #pdb.set_trace()
        #high_iou_row_mask = (iou <= high_thres) & (iou > low_thres)
        high_iou_row_mask = iou <= high_thres
        high_iou_valid_rows = high_iou_row_mask.all(dim=1)
        high_iouvalid_row_indices = torch.nonzero(high_iou_valid_rows).squeeze()

        if low_iouvalid_row_indices.numel() == 0:
            #merged_target = per_gt
            low_iou_pse_target = BoxList(torch.empty(0, 4), per_pse.size, mode="xyxy")
            #low_iou_pse_target.add_field("labels", torch.tensor([]))
        else:
            box1_valid = box1[low_iouvalid_row_indices].view(-1,4)
            low_iou_pse_target = BoxList(box1_valid, per_pse.size)
            #merged_target = BoxList(torch.cat([box2, box1_valid], dim=0), per_gt.size)
            #per_targe_label = per_gt.get_field("labels").view(1,-1)
            per_pseudo_targe_label = per_pse.get_field("labels").view(1,-1)
            per_low_iou_pseudo_targe_label = per_pseudo_targe_label[:,low_iouvalid_row_indices].view(1,-1)
            #per_labels = torch.cat((per_targe_label,per_low_iou_pseudo_targe_label),dim=1)
            low_iou_pse_target.add_field("labels", per_low_iou_pseudo_targe_label.view(-1))
            #merged_target.add_field("labels", per_labels.view(-1))

        if high_iouvalid_row_indices.numel() == 0:
            high_iou_pse_target = BoxList(torch.empty(0, 4), per_pse.size, mode="xyxy")
            #high_iou_pse_target.add_field("labels", torch.tensor([]))
        else:
            box1_valid = box1[high_iouvalid_row_indices].view(-1,4)
            high_iou_pse_target = BoxList(box1_valid, per_pse.size)
            per_pseudo_targe_label = per_pse.get_field("labels").view(1,-1)
            per_high_iou_pseudo_targe_label = per_pseudo_targe_label[:,high_iouvalid_row_indices].view(1,-1)
            high_iou_pse_target.add_field("labels", per_high_iou_pseudo_targe_label.view(-1))

        #all_target += [merged_target]
        low_iou_target += [low_iou_pse_target]
        high_iou_target += [high_iou_pse_target]
        #print("now_target:",all_target)
    #return all_target, low_iou_target, high_iou_target
    return low_iou_target, high_iou_target


def merge_targes(targetsA, targetsB):
    all_targets = []
    for tarA, tarB in zip(targetsA, targetsB):
        if len(tarB) == 0:
            merged_target = tarA
        else:
            merged_target = BoxList(torch.cat([tarA.bbox.cuda(), tarB.bbox.cuda()], dim=0), tarA.size)
            labelA = tarA.get_field("labels").view(1,-1)
            labelB = tarB.get_field("labels").view(1,-1)
            merged_label = torch.cat((labelA,labelB),dim=1)
            merged_target.add_field("labels", merged_label.view(-1))
        all_targets += [merged_target]
    return all_targets

def rpn_merge_targets(targetsA, targetsB):
    all_targets = []
    for tarA, tarB in zip(targetsA, targetsB):
        if len(tarB) == 0:
            merged_target = tarA
        else:
            merged_target = BoxList(torch.cat([tarA.bbox.cuda(), tarB.bbox.cuda()], dim=0), tarA.size)
        all_targets += [merged_target]
    return all_targets


def merge_pseudo_labels(pseudo_targets, gt_targets, low_thres=0.4, high_thres=0.8, low_weight=1.0, high_weight=0.3):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes of (pseudo_targets), sized [N,4].
      box2: (tensor) bounding boxes of (gt_targets), sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    #pdb.set_trace()
    all_merged_targes = []

    for per_pse, per_gt in zip(pseudo_targets, gt_targets):
        # 首先计算两个box左上角点坐标的最大值和右下角坐标的最小值，然后计算交集面积，最后把交集面积除以对应的并集面积
        box1 = per_pse.bbox
        box2 = per_gt.bbox
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(  # 左上角的点
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(  # 右下角的点
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 指两个box没有重叠区域
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2]-box1[:, 0]) * (box1[:, 3]-box1[:, 1])  # [N,]
        area2 = (box2[:, 2]-box2[:, 0]) * (box2[:, 3]-box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        
        low_iou_row_mask = iou <= low_thres
        low_iou_valid_rows = low_iou_row_mask.all(dim=1)
        low_iouvalid_row_indices = torch.nonzero(low_iou_valid_rows).squeeze()

        high_iou_row_mask = (iou <= high_thres) & (iou > low_thres)
        high_iou_valid_rows = high_iou_row_mask.all(dim=1)
        high_iouvalid_row_indices = torch.nonzero(high_iou_valid_rows).squeeze()

        per_targe_label = per_gt.get_field("labels")
        # per_targe_weight = torch.full_like(per_targe_label, 1.0, dtype=torch.float)
        # per_gt.add_field("weights", per_targe_weight)
        per_targe_weight = per_gt.get_field("weights")

        per_pseudo_targe_label = per_pse.get_field("labels").view(1,-1)
        pse_boxes = torch.empty(0, 4, dtype=torch.float).cuda()
        pse_labels = torch.tensor([], dtype=torch.long).cuda()
        pse_weight = torch.tensor([], dtype=torch.float).cuda()

        if high_iouvalid_row_indices.numel() == 0 and low_iouvalid_row_indices.numel() == 0:
            merged_target = per_gt

        else:
            if high_iouvalid_row_indices.numel() > 0:
                box3_valid = box1[high_iouvalid_row_indices].view(-1,4)
                per_high_iou_pseudo_targe_label = per_pseudo_targe_label[:,high_iouvalid_row_indices].view(-1)
                high_iou_pse_target_weight = torch.full_like(per_high_iou_pseudo_targe_label, high_weight, dtype=torch.float)
                pse_boxes = torch.cat([pse_boxes, box3_valid], dim=0)
                pse_labels = torch.cat([pse_labels, per_high_iou_pseudo_targe_label])
                pse_weight = torch.cat([pse_weight, high_iou_pse_target_weight])

            if low_iouvalid_row_indices.numel() > 0:
                box1_valid = box1[low_iouvalid_row_indices].view(-1,4)
                per_low_iou_pseudo_targe_label = per_pseudo_targe_label[:,low_iouvalid_row_indices].view(-1)
                low_iou_pse_target_weight = torch.full_like(per_low_iou_pseudo_targe_label, low_weight, dtype=torch.float)
                pse_boxes = torch.cat([pse_boxes, box1_valid], dim=0)
                pse_labels = torch.cat([pse_labels, per_low_iou_pseudo_targe_label])
                pse_weight = torch.cat([pse_weight, low_iou_pse_target_weight])
            
            merge_boxes = torch.cat([box2, pse_boxes], dim=0)
            merge_labels = torch.cat([per_targe_label, pse_labels])
            merge_weight = torch.cat([per_targe_weight, pse_weight])
            
            merged_target = BoxList(merge_boxes, per_gt.size)
            merged_target.add_field("labels", merge_labels)
            merged_target.add_field("weights", merge_weight)
            
        all_merged_targes += [merged_target]

    return all_merged_targes