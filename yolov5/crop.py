import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm


def read_file(filename):
    with open(filename) as f:
        lines = f.readlines()
    return lines


def norm_cxy2xxyy(norm_cbbox):
    cx, cy, w, h = norm_cbbox
    w *= 1.1
    h *= 1.1
    x1 = max(0, cx - w / 2)
    y1 = max(0, cy - h / 2)
    x2 = min(1, x1 + w)
    y2 = min(1, y1 + h)
    return [x1, y1, x2, y2]


def norm2image_level(norm_bbox, height, width):
    bbox = [norm_bbox[0] * width, norm_bbox[1] * height, norm_bbox[2] * width, norm_bbox[3] * height]
    return [int(x) for x in bbox]


def crop_one_img(pred_txt_name, img_root, crop_img_root, dst_h=320):
    base_name = os.path.basename(pred_txt_name).split(".")[0]
    img_name = "%s/%s.JPG" % (img_root, base_name)
    img = cv2.imread(img_name)
    height, width, _ = img.shape

    os.makedirs(crop_img_root, exist_ok=True)

    preds = read_file(pred_txt_name)
    for idx, pred in enumerate(preds):
        cls_id, cx, cy, w, h = [float(x) for x in pred.split(" ")]
        bbox = norm_cxy2xxyy([cx, cy, w, h])
        x1, y1, x2, y2 = norm2image_level(bbox, height, width)
        crop_img = img[y1: y2 + 1, x1: x2 + 1, :]
        ori_h = h * height
        ratio = dst_h / ori_h
        dst_w = int(w * width * ratio)
        if dst_w < 200:
            continue
        crop_img = cv2.resize(crop_img, (dst_w, int(dst_h)))
        crop_img_name = "%s/%s_%s_%s.JPG" % (crop_img_root, int(cls_id), base_name, str(idx).zfill(3))
        cv2.imwrite(crop_img_name, crop_img)


def crop_batch_img(pred_txt_root, img_root, crop_img_root, new_height=320):
    """
    1. 使用训练图预测，得到预测的归一化bbox
    2. 在原图上crop
    3. 裁剪后的图，resize到新尺寸：pred_norm_h/w * ori_h/ori_w * ratio
    """
    os.makedirs(crop_img_root, exist_ok=True)

    pred_txts = glob("%s/*.txt" % pred_txt_root)
    for pred_txt in tqdm(pred_txts):
        pred_txt = pred_txt.replace("\\", "/")
        crop_one_img(pred_txt, img_root, crop_img_root, new_height)


new_height = 320
pred_txt_root = "runs/detect/bolt/labels"
img_root = "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/bolt/images/train_ori"
crop_img_root = "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/bolt_single/images/train"
crop_batch_img(pred_txt_root, img_root, crop_img_root, new_height)
