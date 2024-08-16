import os
import cv2
import pdb
from PIL import Image
import numpy as np

CATEGORIES = [
    "__background__ ", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

font = cv2.FONT_HERSHEY_SIMPLEX
COLORS = [(np.random.randint(255), np.random.randint(255), np.random.randint(254)) for _ in range(len(CATEGORIES))]

# red color (ground truth targets)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255,0,0)

def draw_image_target(img_id, input_targets):
    imgpath = os.path.join("data/voc07/VOCdevkit/VOC2007", "JPEGImages", "%s.jpg")
    per_img_path = imgpath % img_id
    input_images = Image.open(per_img_path).convert("RGB")

    input_targets = input_targets.resize(input_images.size)

    output_image_path = "/home/qijie/workspace/visual/initial/%s.jpg" % img_id
    img = cv2.cvtColor(np.asarray(input_images), cv2.COLOR_RGB2BGR)

    # draw ground truth targets
    gt_bbox = input_targets.bbox.to("cpu").numpy()
    gt_label = input_targets.get_field("labels").to("cpu").numpy()
    for bbox, label in zip(gt_bbox, gt_label):
        if label >= len(COLORS):
            continue
        left, top, right, bottom = bbox.astype(int)
        cv2.rectangle(img, (left, top), (right, bottom), RED_COLOR, 2)
        cv2.putText(img, f'{CATEGORIES[label]}', (left, max(top - 5, 0)), font, 0.5, RED_COLOR, 1)

    cv2.imwrite(output_image_path, img)

def draw_singe_image(img_id, input_targets, del_proposals, add_proposals=None):
    
    imgpath = os.path.join("data/voc07/VOCdevkit/VOC2007", "JPEGImages", "%s.jpg")

    per_img_path = imgpath % img_id
    input_images = Image.open(per_img_path).convert("RGB")

    input_targets = input_targets.resize(input_images.size)
    del_proposals = del_proposals.resize(input_images.size)
    # add_proposals = add_proposals.resize(input_images.size)

    output_image_path = "/home/qijie/workspace/visual/rpn/%s.jpg" % img_id
    img = cv2.cvtColor(np.asarray(input_images), cv2.COLOR_RGB2BGR)

    # 绘制 del_proposals
    del_bbox = del_proposals.bbox.to("cpu").numpy()
    for bbox in del_bbox:
        left, top, right, bottom = bbox.astype(int)
        cv2.rectangle(img, (left, top), (right, bottom), GREEN_COLOR, 2)

    # add_bbox = add_proposals.bbox.to("cpu").numpy()
    # for bbox in add_bbox:
    #     left, top, right, bottom = bbox.astype(int)
    #     cv2.rectangle(img, (left, top), (right, bottom), BLUE_COLOR, 2)

    # draw ground truth targets
    gt_bbox = input_targets.bbox.to("cpu").numpy()
    gt_label = input_targets.get_field("labels").to("cpu").numpy()
    for bbox, label in zip(gt_bbox, gt_label):
        if label >= len(COLORS):  # 跳过不在CATEGORIES定义中的类别
            continue
        left, top, right, bottom = bbox.astype(int)
        cv2.rectangle(img, (left, top), (right, bottom), RED_COLOR, 2)
        cv2.putText(img, f'{CATEGORIES[int(label)]}', (left, max(top - 5, 0)), font, 0.5, RED_COLOR, 1)

    cv2.imwrite(output_image_path, img)

def Draw_singe_image(img_id, input_images, input_targets, del_proposals, round):
    
    input_images = input_images.cpu().numpy().transpose(1, 2, 0)
    
    #output_ini_image_path = "/home/qijie/workspace/visual/ini/%s.jpg" % img_id
    #cv2.imwrite(output_ini_image_path, input_images)
    # 将图像数据转换到0-255，并转换为uint8
    # input_images = np.clip(input_images, 0, 1)
    # input_images = (input_images * 255).astype(np.uint8)
    img_id = round + '_' + img_id
    output_image_path = "/home/qijie/workspace/visual/rpn_att/%s.jpg" % img_id
    img = cv2.cvtColor(np.asarray(input_images), cv2.COLOR_RGB2BGR)

    # 绘制 del_proposals
    del_bbox = del_proposals.bbox.to("cpu").numpy()
    for bbox in del_bbox:
        left, top, right, bottom = bbox.astype(int)
        cv2.rectangle(img, (left, top), (right, bottom), GREEN_COLOR, 2)

    # 绘制 ground truth targets
    gt_bbox = input_targets.bbox.to("cpu").numpy()
    gt_label = input_targets.get_field("labels").to("cpu").numpy()
    for bbox, label in zip(gt_bbox, gt_label):
        if label >= len(COLORS):  # 跳过不在CATEGORIES定义中的类别
            continue
        left, top, right, bottom = bbox.astype(int)
        cv2.rectangle(img, (left, top), (right, bottom), RED_COLOR, 2)
        cv2.putText(img, f'{CATEGORIES[int(label)]}', (left, max(top - 5, 0)), font, 0.5, RED_COLOR, 1)

    cv2.imwrite(output_image_path, img)