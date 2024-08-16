#!/bin/bash
port=$(python get_free_port.py)
GPU=1

alias exp="CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --master_port=${port} --nproc_per_node=${GPU} tools/train_incremental_finetune_all.py"
shopt -s expand_aliases

task=10-10
exp -t ${task} -n test --cls 0.15 -l 0.4 -high 0.7 -lw 1.0 -hw 0.3