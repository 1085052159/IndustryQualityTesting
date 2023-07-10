"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import os
import sys
import time
from glob import glob
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from utils.augmentations import letterbox

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device


def read_imgs(img_names):
    if isinstance(img_names, str):
        if os.path.isdir(img_names):
            root_path = img_names
            img_names = glob("%s/*.JPG" % root_path)
        if os.path.isfile(img_names):
            if img_names.endswith(".txt"):
                with open(img_names) as f:
                    img_names = f.readlines()
            else:
                img_names = [img_names]

    imgs = []
    new_img_names = []
    for img_name in img_names:
        img = cv2.imread(img_name)
        new_img_names.append(img_name)
        imgs.append(img)
    return imgs, new_img_names


def preprocess_imgs(imgs, img_size, stride):
    """
    :param imgs: a list, each element is an image array, h * w * c
    :return:
    """
    assert type(imgs) == list
    imgs_array = []
    # import pdb
    # pdb.set_trace()
    for i in range(len(imgs)):
        img = letterbox(imgs[i], img_size, stride=stride)[0]
        if len(imgs_array) != 0:
            h, w = imgs_array[0].shape[1:]
        else:
            h, w = img.shape[: 2]
        img = cv2.resize(img, (w, h))
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        imgs_array.append(img)
    imgs_array = np.array(imgs_array)
    return imgs_array


def vis_det_results(det_results, save_img_names, line_thickness=3):
    names = det_results["names"]
    imgs = det_results["input_imgs"]
    preds = det_results["preds"]
    assert len(imgs) == len(preds)
    assert len(imgs) == len(save_img_names)
    for j in range(len(imgs)):
        im1 = imgs[j].copy()
        img_h, img_w, _ = im1.shape
        pred = preds[j].copy()
        save_img_name = save_img_names[j]
        if len(pred) == 0:
            continue
        pred[:, 1] *= img_w
        pred[:, 3] *= img_w
        pred[:, 2] *= img_h
        pred[:, 4] *= img_h
        pred[:, 1:5] = xywh2xyxy(pred[:, 1:5])
        for i, det in enumerate(pred):
            cls, *xyxy, conf = det
            c = int(cls)  # integer class
            label = "%s %.2f" % (names[c], conf)
            plot_one_box(xyxy, im1, label=label, color=colors(c, True), line_thickness=line_thickness)
        cv2.imwrite(save_img_name, im1)


def vis_det_center_results(det_results, save_img_names, line_thickness=3):
    names = det_results["names"]
    imgs = det_results["input_imgs"]
    preds = det_results["preds"]
    assert len(imgs) == len(preds)
    assert len(imgs) == len(save_img_names)
    for j in range(len(imgs)):
        im1 = imgs[j].copy()
        img_h, img_w, _ = im1.shape
        pred = preds[j].copy()
        save_img_name = save_img_names[j]
        if len(pred) == 0:
            continue
        pred[:, 1] *= img_w
        pred[:, 3] *= img_w
        pred[:, 2] *= img_h
        pred[:, 4] *= img_h
        pred[:, 1:5] = xywh2xyxy(pred[:, 1:5])
        for i, det in enumerate(pred):
            cls, *xyxy, conf = det
            c = int(cls)  # integer class
            label = "%s %.2f" % (names[c], conf)
            plot_one_box(xyxy, im1, label=label, color=colors(c, True), line_thickness=line_thickness)
        cv2.imwrite(save_img_name, im1)


def init_model_detector(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


@torch.no_grad()
def run_detector(model,  # model.pt path(s)
                 source=[],  # list, each element is img array with shape (h, w, c)
                 imgsz=640,  # inference size (pixels)
                 conf_thres=0.25,  # confidence threshold
                 iou_thres=0.45,  # NMS IOU threshold
                 max_det=1000,  # maximum detections per image
                 device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 classes=None,  # filter by class: --class 0, or --class 0 2 3
                 agnostic_nms=False,  # class-agnostic NMS
                 augment=False,  # augmented inference
                 half=False,  # use FP16 half-precision inference
                 ):
    results = {}
    results["input_imgs"] = source
    # each element is an one image results, [[norm_cx, norm_cy, norm_w, norm_h, cls]]
    results["preds"] = []

    # Load model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    results["names"] = names

    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()  # to FP16

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    imgs_numpy = preprocess_imgs(source, imgsz, stride)
    img = torch.from_numpy(imgs_numpy).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    # Inference
    preds = model(img,
                  augment=augment,
                  visualize=False)[0]

    # Apply NMS
    preds = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    # preds: list, each element is [n, 6], cls, xywh, conf
    for j, pred in enumerate(preds):
        im0 = source[j]
        res = np.zeros((pred.shape[0], 6), dtype=np.float32)
        if len(pred) > 0:
            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0.shape).round()
            gn = np.array(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            pred = pred.cpu().numpy()

            xywh = xyxy2xywh(pred[:, :4] / gn)

            res[:, 0] = pred[:, -1]  # cls
            res[:, -1] = pred[:, -2]  # conf
            res[:, 1: 5] = xywh
        results["preds"].append(res)
    return results


def main():
    device = "cuda"
    ckpt_det = "./runs/train/piezometer/weights/best.pt"
    save_path = "tmp_results_debug"
    img_names = [
        "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/piezometer/images/train/DJI_20230525110017_0077_V.JPG",
        "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/piezometer/images/train/DJI_20230525105404_0038_V.JPG"
    ]

    # ckpt_det = "./runs/train/bolt/weights/best.pt"
    # save_path = "tmp_results_debug1"
    # img_names = [
    #     "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/bolt/images/train/DJI_20230613103035_0004_V.JPG",
    #     "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/bolt/images/train/DJI_20230613103430_0035_V.JPG"
    # ]
    source, _ = read_imgs(img_names)
    ##########################################
    # init piezometer detector
    ##########################################
    model_det = init_model_detector(ckpt_det, device)
    preds = run_detector(model_det,  # model.pt path(s)
                         source=source,  # file/dir/URL/glob, 0 for webcam
                         imgsz=640,  # inference size (pixels)
                         conf_thres=0.25,  # confidence threshold
                         iou_thres=0.45,  # NMS IOU threshold
                         max_det=1000,  # maximum detections per image
                         device=device,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                         )
    save_img_names = ["%s/%s" % (save_path, os.path.basename(img_name)) for img_name in img_names]
    os.makedirs(save_path, exist_ok=True)
    vis_det_results(preds, save_img_names, line_thickness=3)
    print(preds["names"])
    print(preds["preds"])


if __name__ == "__main__":
    main()
