import json
import os
import random
import shutil

import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

random.seed(123456)


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


def gen_masks(json_path):
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


def split_train_test(img_root, train_ratio=0.8):
    img_abso_paths = glob("%s/*" % (img_root))
    img_abso_paths = [x.replace("\\", "/") for x in img_abso_paths]
    for src_img_path in tqdm(img_abso_paths):
        src_labels_json_path = "%s.json" % src_img_path.replace("/images/", "/labels_json/").split(".")[0]
        src_masks_path = "%s.png" % src_img_path.replace("/images/", "/masks/").split(".")[0]
        src_masks_ori_path = "%s.png" % src_img_path.replace("/images", "/masks_ori/").split(".")[0]
        if random.random() <= train_ratio:
            dst_img_path = src_img_path.replace("/images/", "/images/train/")
            dst_labels_json_path = src_labels_json_path.replace("/labels_json/", "/labels_json/train/")
            dst_masks_path = src_masks_path.replace("/masks/", "/masks/train/")
            dst_masks_ori_path = src_masks_ori_path.replace("/masks_ori/", "/masks_ori/train/")
        else:
            dst_img_path = src_img_path.replace("/images/", "/images/val/")
            dst_labels_json_path = src_labels_json_path.replace("/labels_json/", "/labels_json/val/")
            dst_masks_path = src_masks_path.replace("/masks/", "/masks/val/")
            dst_masks_ori_path = src_masks_ori_path.replace("/masks_ori/", "/masks_ori/val/")
        os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
        os.makedirs(os.path.dirname(dst_labels_json_path), exist_ok=True)
        os.makedirs(os.path.dirname(dst_masks_path), exist_ok=True)
        os.makedirs(os.path.dirname(dst_masks_ori_path), exist_ok=True)
        
        if os.path.exists(src_img_path):
            shutil.move(src_img_path, dst_img_path)
        if os.path.exists(src_labels_json_path):
            shutil.move(src_labels_json_path, dst_labels_json_path)
        if os.path.exists(src_masks_path):
            shutil.move(src_masks_path, dst_masks_path)
        if os.path.exists(src_masks_ori_path):
            shutil.move(src_masks_ori_path, dst_masks_ori_path)


def jpg2png(base_root, save_root):
    img_paths = glob("%s/*/**" % base_root)
    # import pdb
    # pdb.set_trace()
    for img_path in tqdm(img_paths):
        img_path = img_path.replace("\\", "/")
        if not img_path.endswith(".png"):
            img = cv2.imread(img_path)
            new_img_path = "%s/%s.png" % (save_root, os.path.basename(img_path).split(".")[0])
            cv2.imwrite(new_img_path, img)
            os.remove(img_path)
        


json_path = "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/bolt_single/labels_json/train"
json_path = "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/bolt"
json_path = "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/bolt_single1/labels_json/train"
json_path = "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/bolt_single_total/labels_json/"
# gen_masks(json_path)
img_root = "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/bolt_single_total/images"
# jpg2png(img_root, img_root)
split_train_test(img_root, 0.9)

