# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os
import cv2
import torch
from tqdm import tqdm
from PIL import Image
from maskrcnn_benchmark.data import datasets
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..structures.segmentation_mask import SegmentationMask
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
import numpy as np
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def compute_on_dataset(model, data_loader, device, timer=None, external_proposal=False, summary_writer=None):
    NAME_CLASSES = np.array(["#bkg","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                       "tvmonitor"])
    model.eval()
    conf_matrix = torch.zeros((len(NAME_CLASSES), len(NAME_CLASSES)))
    # BACK_CONF_THRESHOLD = 0.3
    results_dict = {}
    results_background_dict = {}
    cpu_device = torch.device("cpu")
    dataset = data_loader.dataset
    # num_back_box = 0
    # tot_bbox_under_conf = 0
    for idx, batch in enumerate(tqdm(data_loader)):
        images, targets, proposals, img_id, index = batch
        ious = []
        # load images and proposals to gpu
        images = images.to(device)
        if external_proposal:
            proposals = [proposal.to(device) for proposal in proposals]
        with torch.no_grad():
            if timer:
                timer.tic()
            if external_proposal:  # use external proposals
                output = model.use_external_proposals_edgeboxes(images, proposals)
            else:
                targets = [target.to(device) for target in targets]
                #output, features, results_background = model(images, targets)
                output, features, results_background = model(images)

                #draw_image(img_id, targets, output)
                #visual_prediction(output, img_id)

            if timer:
                torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]

        # if summary_writer and idx % 30 == 0:
        #     good_bbox_idx = (output[0].extra_fields["scores"] > 0.2).nonzero().squeeze()
        #     if good_bbox_idx.numel() == 1:
        #         good_bbox_idx.unsqueeze()
        #     if good_bbox_idx.numel() > 0:
        #         string_labels = list(NAME_CLASSES[output[0].extra_fields["labels"]][good_bbox_idx])
        #         summary_writer.add_image_with_boxes(str(idx), img_vis[0], output[0].bbox[good_bbox_idx],
        #                                         labels=string_labels)

        results_dict.update({idx: result for idx, result in zip(index, output)})
        results_background_dict.update(({idx: results_background for idx, result in zip(index, [results_background])}))
    #     add_matrix = test_background_fall(dataset, idx, output[0], results_background, len(NAME_CLASSES))
    #     conf_matrix += add_matrix
    # print(conf_matrix)
    # with open("conf_matrix_FILOD_UCE_UKDx10_15_15.txt", "w") as f:
    #     print(conf_matrix, file=f)
    return results_dict, results_background_dict

def test_background_fall(dataset, idx, results, results_background, n_classes):
    add_matrix = torch.zeros((n_classes, n_classes))
    gt_boxlist = dataset.get_groundtruth(idx).to(results_background.bbox.device)
    results.bbox = results.bbox.to(results_background.bbox.device)
    img_info = dataset.get_img_info(idx)
    image_width = img_info["width"]
    image_height = img_info["height"]
    results_background = results_background.resize((image_width, image_height))
    results = results.resize((image_width, image_height))
    ious_back = boxlist_iou(gt_boxlist, results_background)
    back_pred_wrong = (ious_back > 0.5).nonzero(as_tuple=True)
    ious_pred = boxlist_iou(gt_boxlist, results)
    pred_ious = (ious_pred > 0.5).nonzero(as_tuple=True)
    print("DAJEE")

    add_matrix[0][0] += len(results_background) - len(back_pred_wrong[1])
    for rbw in back_pred_wrong[0]:
        add_matrix[gt_boxlist.extra_fields["labels"][rbw]][0] += 1

    for index, r in enumerate(results.extra_fields["labels"]):
        matched = pred_ious[1] == index
        true_labels = gt_boxlist.extra_fields["labels"][pred_ious[0][matched]]

        if len(true_labels) == 0:
            add_matrix[0][r.item()] += 1
        elif r.item() in true_labels:
            add_matrix[r.item(), r.item()] += 1
        else:
            add_matrix[true_labels[0], r.item()] += 1

    return add_matrix


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = [predictions_per_gpu]  # all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark_target_model.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(model, data_loader, dataset_name, iou_types=("bbox",), box_only=False, device="cuda",
        expected_results=(), expected_results_sigma_tol=4, output_folder=None, external_proposal=False,
              alphabetical_order=True, summary_writer=None, save_predictions=False):

    # ONLY SUPPORTED ON SINGLE GPU!
    print('inference.py | alphabetical_order: {0}'.format(alphabetical_order))
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark_target_model.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions, back_predictions = compute_on_dataset(model, data_loader, device, inference_timer, external_proposal, summary_writer)
    #pdb.set_trace()
    # wait for all processes to complete before measuring the time
    # synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)

    if not is_main_process():
        return

    if output_folder and save_predictions:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    print('inference.py | alphabetical_order: {0}'.format(alphabetical_order))
    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
        alphabetical_order=alphabetical_order
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)





# 类别列表
CATEGORIES = [
    "__background__ ", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# OpenCV字体
font = cv2.FONT_HERSHEY_SIMPLEX

# 为每个类别分配一个唯一的颜色 (B, G, R)
COLORS = [(np.random.randint(255), np.random.randint(255), np.random.randint(254)) for _ in range(len(CATEGORIES))]

# 红色框用于标注ground truth targets
RED_COLOR = (0, 0, 255)

def draw_image(all_img_id, all_input_targets, all_input_proposals):
    imgpath = os.path.join("data/voc07/VOCdevkit/VOC2007", "JPEGImages", "%s.jpg")
    #pdb.set_trace()
    for img_id, input_targets, input_proposals in zip(all_img_id, all_input_targets, all_input_proposals):
        per_img_path = imgpath % img_id
        input_images = Image.open(per_img_path).convert("RGB")

        input_targets = input_targets.resize(input_images.size)
        input_proposals = input_proposals.resize(input_images.size)

        output_image_path = "/home/qijie/workspace/visual/%s.jpg" % img_id
        img = cv2.cvtColor(np.asarray(input_images), cv2.COLOR_RGB2BGR)

        # 绘制 proposals
        proposal_bbox = input_proposals.bbox.to("cpu").numpy()
        proposal_label = input_proposals.get_field("labels").to("cpu").numpy()
        for bbox, label in zip(proposal_bbox, proposal_label):
            color = COLORS[label]
            left, top, right, bottom = bbox.astype(int)
            cv2.rectangle(img, (left, top), (right, bottom), color, 2)
            cv2.putText(img, f'{CATEGORIES[label]}', (left, max(top - 5, 0)), font, 0.5, color, 1)

        # 绘制 ground truth targets
        gt_bbox = input_targets.bbox.to("cpu").numpy()
        gt_label = input_targets.get_field("labels").to("cpu").numpy()
        for bbox, label in zip(gt_bbox, gt_label):
            if label >= len(COLORS):  # 跳过不在CATEGORIES定义中的类别
                continue
            left, top, right, bottom = bbox.astype(int)
            cv2.rectangle(img, (left, top), (right, bottom), RED_COLOR, 2)
            cv2.putText(img, f'{CATEGORIES[label]}', (left, max(top - 5, 0)), font, 0.5, RED_COLOR, 1)

        cv2.imwrite(output_image_path, img)



def visual_prediction(predicitons, img_ids):
    for i in range(len(img_ids)):
        img_path = os.path.join("data/voc07/VOCdevkit/VOC2007", "JPEGImages", "{:06d}.jpg".format(img_ids[i]))
        image = cv2.imread(img_path)
        result = image.copy()
        result = overlay_boxes(result, predicitons)
        result = overlay_class_names(result, predicitons)

def compute_colors_for_labels(labels):
    """
    Simple function that adds fixed colors depending on the class
    """
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors

def overlay_boxes(image, predictions):
    """
    Adds the predicted boxes on top of the image

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    labels = predictions.get_field("labels")
    boxes = predictions.bbox

    colors = compute_colors_for_labels(labels).tolist()

    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color), 1
        )

    return image

def overlay_class_names(image, predictions):
    """
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    """
    CATEGORIES = [
        "__background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog","horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
    scores = predictions.get_field("scores").tolist()
    labels = predictions.get_field("labels").tolist()
    labels = [CATEGORIES[i] for i in labels]
    boxes = predictions.bbox

    template = "{}: {:.2f}"
    for box, score, label in zip(boxes, scores, labels):
        x, y = box[:2]
        s = template.format(label, score)
        cv2.putText(
            image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
        )
    return image