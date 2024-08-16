# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from torch.utils.tensorboard import SummaryWriter

from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir


# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "-c", "--config-file",
        default="/home/ageraci/FILOD/Faster-ILOD/configs/IS_cfg/e2e_mask_rcnn_R_50_C4_1x_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "-w", "--weight",
        default="NONE",
        metavar="FILE",
        help="path to config file",
    )
    args = parser.parse_args()

    num_gpus = 1
    distributed = num_gpus > 1

    if args.local_rank != 0:
        return

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.weight is not "NONE":
        cfg.MODEL.WEIGHT = args.weight
    
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH = 0.5
    #cfg.MODEL.ROI_HEADS.NMS = 0.3
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16'
    amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

    output_dir = cfg.OUTPUT_DIR
    weight_name = cfg.MODEL.WEIGHT
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    summary_writer = SummaryWriter(log_dir=cfg.TENSORBOARD_DIR)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        result = inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            alphabetical_order=cfg.TEST.COCO_ALPHABETICAL_ORDER,
            summary_writer=summary_writer
        )
        ap_old = result["ap"][1:len(cfg.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES)+1].mean()
        ap_new = result["ap"][len(cfg.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES)+1:1+len(cfg.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES)+len(cfg.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES)].mean()
        ap_all = result["ap"][1:1+len(cfg.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES)+len(cfg.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES)].mean()
        print("ap_old:",ap_old, "ap_new:",ap_new, "ap_all:",ap_all)
        with open(os.path.join("results", f"result.txt"), "a") as fid:
            fid.write(f"{weight_name}, ap_old:{ap_old}, ap_new:{ap_new} , ap_all:{ap_all}\n")
        synchronize()

if __name__ == "__main__":
    main()
