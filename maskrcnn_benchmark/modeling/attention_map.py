import torch 
from torch import nn
import torch.nn.functional as F
from torchvision.ops import roi_align

def generate_attention_map(features):
    """
    Args:
        features(Tensor): Bs*C*H*W, feature map
    """
    temp = 2
    features = features[0].detach()
    attention_maps = activation_at(features, temp)
    return attention_maps


def add_attention_scores(proposals, attention_maps):
    for img_idx, proposal in enumerate(proposals):
        attention_map = attention_maps[img_idx].unsqueeze(0).unsqueeze(0)  # 假设attention_map是单通道的
        img_size = proposal.size
        W_img, H_img = img_size
        H_fea, W_fea = attention_map.shape[2], attention_map.shape[3]

        # resize the shape (proposal -> feature map)
        proposals_boxes = proposal.bbox.clone()
        proposals_boxes[:, [0, 2]] *= W_fea / W_img
        proposals_boxes[:, [1, 3]] *= H_fea / H_img

        attention_scores = roi_align(attention_map, [proposals_boxes], output_size=(7, 7)).mean([2, 3]).squeeze(1)

        proposal.add_field("attention", attention_scores)
    return proposals


def calculate_attention_scores_per_img_roi_align(proposals, attention_map):
    img_size = proposals.size
    W_img, H_img = img_size
    H_fea, W_fea = attention_map.shape[0], attention_map.shape[1]
    proposals_boxes = proposals.bbox

    proposals_boxes[:, 0] = proposals_boxes[:, 0]*W_fea/W_img
    proposals_boxes[:, 1] = proposals_boxes[:, 1]*H_fea/H_img
    proposals_boxes[:, 2] = proposals_boxes[:, 2]*W_fea/W_img
    proposals_boxes[:, 3] = proposals_boxes[:, 3]*H_fea/H_img

    # select 7*7 regions to get mean attention scores
    attention_scores = roi_align(attention_map.unsqueeze(0).unsqueeze(0), [proposals_boxes], [7,7]).mean(3).mean(2)

    return attention_scores


def activation_at(f_map, temp=2):
    N, C, H, W = f_map.shape

    value = torch.abs(f_map)
    # Bs*W*H
    fea_map = value.pow(temp).mean(axis=1, keepdim=True)
    attention_maps = (H * W * F.softmax(fea_map.view(N, -1), dim=1)).view(N, H, W)
    
    return attention_maps
