#!/bin/bash
port=$(python get_free_port.py)
GPU=1

# SECONDE STEP

# sec 10
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --master_port=${port} --nproc_per_node=${GPU} tools/train_first_step.py -c configs/OD_cfg/10-10/e2e_faster_rcnn_R_50_C4_8x_sec10.yaml


# sec 5
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --master_port=${port} --nproc_per_node=${GPU} tools/train_first_step.py -c configs/OD_cfg/15-5/e2e_faster_rcnn_R_50_C4_8x_sec5.yaml
