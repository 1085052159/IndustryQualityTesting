import json
import os

import numpy as np
import cv2
from glob import glob

from tqdm import tqdm


def convertPolygonToMask(json_file_path):
    with open(json_file_path, "r", encoding='utf-8') as f:
        data = json.load(f)
        img_h = data["imageHeight"]
        img_w = data["imageWidth"]
        mask = np.zeros((img_h, img_w), np.uint8)
        # 图片中目标的数量 num=len(data["shapes"])
        num = 0
        for obj in data["shapes"]:
            label = obj["label"]
            polygonPoints = obj["points"]
            polygonPoints = np.array(polygonPoints, np.int32)
            # print("+" * 50, "\n", polygonPoints)
            # print(label)
            num += 1
            cv2.drawContours(mask, [polygonPoints], -1, (255), -1)
    
    return mask


json_path = "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/bolt_single/labels_json/train"
json_path = "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/bolt"
json_names = glob("%s/*.json" % json_path)
for json_name in tqdm(json_names):
    mask_ori_save_path = json_name.replace("/labels_json/", "/masks_ori/").replace(".json", ".png")
    os.makedirs(os.path.dirname(mask_ori_save_path), exist_ok=True)
    mask = convertPolygonToMask(json_name)
    cv2.imwrite(mask_ori_save_path, mask)
    mask_save_path = json_name.replace("/labels_json/", "/masks/").replace(".json", ".png")
    os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
    idx = np.where(mask == 255)
    mask[idx] = 1
    cv2.imwrite(mask_save_path, mask)
