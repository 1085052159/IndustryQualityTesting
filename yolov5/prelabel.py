"""
根据预测结果生成labelme/labelbee标注下的json文件
"""
import json
import os
import random
import string
from glob import glob

import cv2
from tqdm import tqdm

from detect_infer import init_model_detector, run_detector


def read_img(img_name):
    img = cv2.imread(img_name)
    if img is None:
        print("not found: %s" % img_name)
    return img, os.path.basename(img_name).split(".")[0]


def resize_img(img, ratio=-1.0):
    if 0. < ratio < 1.:
        h, w, _ = img.shape
        new_h = int(h * ratio)
        new_w = int(w * ratio)
        img = cv2.resize(img, (new_w, new_h))
    return img


def norm_cxy2xxyy(norm_cbbox, scale_coef=1.1):
    cx, cy, w, h = norm_cbbox
    w *= scale_coef
    h *= scale_coef
    x1 = max(0, cx - w / 2)
    y1 = max(0, cy - h / 2)
    x2 = min(1, x1 + w)
    y2 = min(1, y1 + h)
    return [x1, y1, x2, y2]


def norm2image_level(norm_bbox, height, width):
    bbox = [norm_bbox[0] * width, norm_bbox[1] * height, norm_bbox[2] * width, norm_bbox[3] * height]
    return bbox


def get_labelme_format_json(preds, ori_h, ori_w, source):
    labelme_format = {"version": "5.2.1", "flags": {}, "shapes": [
    ], "imagePath": "", "imageData": None, "imageHeight": ori_h, "imageWidth": ori_w}
    labelme_format["imagePath"] = "../%s" % ("/".join(source.split("/")[-2:]))
    classes = preds["names"]
    for pred in preds["preds"][0]:
        cls_id, norm_cx, norm_cy, norm_w, norm_h = pred[: -1]
        cls_id = int(cls_id)
        norm_bbox = norm_cxy2xxyy([norm_cx, norm_cy, norm_w, norm_h], scale_coef=1.0)
        x1, y1, x2, y2 = norm2image_level(norm_bbox, ori_h, ori_w)
        shape = {
            "label": classes[cls_id],
            "points": [
                [x1, y1],
                [x2, y2]
            ],
            "group_id": "",
            "description": "",
            "shape_type": "rectangle",
            "flags": {},
        }
        labelme_format["shapes"].append(shape)
    return labelme_format


def get_labelbee_format_json(preds, ori_h, ori_w):
    labelbee_format = {
        "width": ori_w,
        "height": ori_h,
        "valid": "true",
        "rotate": 0,
        "step_1": {
            "toolName": "rectTool",
            "result": [],
        }
    }
    classes = sorted(preds["names"])
    cnt = 0
    for pred in preds["preds"][0]:
        cnt += 1
        cls_id, norm_cx, norm_cy, norm_w, norm_h = pred[: -1]
        cls_id = int(cls_id)
        norm_bbox = norm_cxy2xxyy([norm_cx, norm_cy, norm_w, norm_h], scale_coef=1.0)
        x1, y1, x2, y2 = norm2image_level(norm_bbox, ori_h, ori_w)
        id_ = ''.join(random.sample(string.ascii_letters + string.digits, 8))
        one_res = {
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1,
            "attribute": str(cls_id),
            "valid": True,
            "id": id_,
            "sourceID": "",
            "textAttribute": "",
            "order": cnt
        }
        labelbee_format["step_1"]["result"].append(one_res)
    return labelbee_format


def write_json(json_name, jsons):
    json_data = json.dumps(jsons, indent=4)
    with open(json_name, "w") as f:
        f.write(json_data)


def prelabel_labelme_format(img_root_path, ckpt_det, device, json_save_path):
    abso_img_paths = glob("%s/*" % img_root_path)
    model_det = init_model_detector(ckpt_det, device)
    LABEL_MAP = {}
    for source in tqdm(abso_img_paths):
        source = source.replace("\\", "/")
        img0, filename = read_img(source)
        ori_h, ori_w, _ = img0.shape
        # imgsz = 320
        imgsz = 640
        img = resize_img(img0, ratio=(imgsz / ori_h))
        preds = run_detector(model_det,  # model.pt path(s)
                             source=[img],  # file/dir/URL/glob, 0 for webcam
                             imgsz=imgsz,  # inference size (pixels)
                             conf_thres=0.25,  # confidence threshold
                             iou_thres=0.2,  # NMS IOU threshold
                             max_det=1000,  # maximum detections per image
                             device=device,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                             )
        names = preds["names"]
        for idx, name in enumerate(names):
            LABEL_MAP[name] = idx
        labelme_format = get_labelme_format_json(preds, ori_h, ori_w, source)
        write_json("%s/%s.json" % (json_save_path, filename), labelme_format)
    write_json("%s/classes.txt" % (json_save_path), LABEL_MAP)


def prelabel_labelbee_format(img_root_path, ckpt_det, device, json_save_path):
    os.makedirs(json_save_path)
    abso_img_paths = glob("%s/*" % img_root_path)
    model_det = init_model_detector(ckpt_det, device)
    LABEL_MAP = {}
    for source in tqdm(abso_img_paths):
        source = source.replace("\\", "/")
        img0, filename = read_img(source)
        ori_h, ori_w, _ = img0.shape
        # imgsz = 320
        imgsz = 640
        img = resize_img(img0, ratio=(imgsz / ori_h))
        preds = run_detector(model_det,  # model.pt path(s)
                             source=[img],  # file/dir/URL/glob, 0 for webcam
                             imgsz=imgsz,  # inference size (pixels)
                             conf_thres=0.25,  # confidence threshold
                             iou_thres=0.2,  # NMS IOU threshold
                             max_det=1000,  # maximum detections per image
                             device=device,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                             )
        names = sorted(preds["names"])
        for idx, name in enumerate(names):
            LABEL_MAP[idx] = name
        labelbee_format = get_labelbee_format_json(preds, ori_h, ori_w)
        write_json("%s/%s.json" % (json_save_path, os.path.basename(source)), labelbee_format)
    write_json("%s/classes_labelbee.txt" % (json_save_path), LABEL_MAP)


img_root_path = "E:/Downloads/annotated_image/bolt/images"
ckpt_det = "./runs/train/bolt/weights/best.pt"
json_save_path = "E:/Downloads/annotated_image/bolt/labels_json"

img_root_path = "F:/dataset/bolt_piezometer/bolt2/test_imgs"
ckpt_det = "./runs/train/bolt_prelabel_yolov5m_new_anchor_640_640_300_82/weights/best.pt"
json_save_path = "F:/dataset/bolt_piezometer/bolt2/test_imgs/labels_json"

# img_root_path = "E:/Downloads/annotated_image/piezometer/images"
# ckpt_det = "./runs/train/piezometer/weights/best.pt"
# json_save_path = "E:/Downloads/annotated_image/piezometer/labels_json"
device = "cuda"
# prelabel_labelme_format(img_root_path, ckpt_det, device, json_save_path)
prelabel_labelbee_format(img_root_path, ckpt_det, device, json_save_path)
