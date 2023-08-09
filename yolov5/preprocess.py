import os
import random
from glob import glob
import json

import cv2
from tqdm import tqdm

random.seed(123456)


def labelbee_json2txt(json_label_root):
    def resolve_labelbee_json(labelbee_json_path):
        with open(labelbee_json_path, "r") as f:
            data = json.load(f)
        img_width = int(data["width"])
        img_height = int(data["height"])
        results = data["step_1"]["result"]
        bboxes = []
        for result in results:
            cls_id = result["attribute"]
            if len(cls_id) == 0:
                continue
            x = float(result["x"])
            y = float(result["y"])
            width = float(result["width"])
            height = float(result["height"])
            cx = x + width / 2
            cy = y + height / 2
            norm_cx = cx / img_width
            norm_cy = cy / img_height
            norm_w = width / img_width
            norm_h = height / img_height
            # bboxes.append([cls_id, norm_cx, norm_cy, norm_w, norm_h])
            bboxes.append([str(cls_id), str(norm_cx), str(norm_cy), str(norm_w), str(norm_h)])
        return bboxes

    json_labels = glob("%s/*.json" % json_label_root)
    for label_path in tqdm(json_labels):
        label_path = label_path.replace("\\", "/")
        if "classes" in label_path.lower():
            continue
        label_txt_path = "%s.txt" % (label_path.split(".")[0].replace("/labels_json/", "/labels/"))
        label_txt_root = os.path.dirname(label_txt_path)
        os.makedirs(label_txt_root, exist_ok=True)
        bboxes = resolve_labelbee_json(label_path)
        with open(label_txt_path, "w") as f:
            f.write("\n".join([" ".join(bbox) for bbox in bboxes]))


def labelme_json2txt(json_label_root):
    def resolve_labelme_json(labelme_json_path):
        with open(labelme_json_path, "r") as f:
            data = json.load(f)
        img_width = int(data["imageWidth"])
        img_height = int(data["imageHeight"])
        shapes = data["shapes"]
        bboxes = []
        for shape in shapes:
            label = shape["label"]
            cls_id = LABEL_MAP[label.lower()]
            points = shape["points"]
            x1, y1 = [float(t) for t in points[0]]
            x2, y2 = [float(t) for t in points[1]]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            norm_cx = cx / img_width
            norm_cy = cy / img_height
            norm_w = w / img_width
            norm_h = h / img_height
            # bboxes.append([cls_id, norm_cx, norm_cy, norm_w, norm_h])
            bboxes.append([str(cls_id), str(norm_cx), str(norm_cy), str(norm_w), str(norm_h)])
        return bboxes

    json_labels = glob("%s/*.json" % json_label_root)
    for label_path in tqdm(json_labels):
        if "classes" in label_path.lower():
            continue
        label_path = label_path.replace("\\", "/")
        label_txt_path = "%s.txt" % (label_path.split(".")[0].replace("/labels_json/", "/labels/"))
        label_txt_root = os.path.dirname(label_txt_path)
        os.makedirs(label_txt_root, exist_ok=True)
        bboxes = resolve_labelme_json(label_path)
        with open(label_txt_path, "w") as f:
            f.write("\n".join([" ".join(bbox) for bbox in bboxes]))


def verify(json_label_root):
    label_json_path = glob("%s/*.json" % json_label_root)[10].replace("\\", "/")
    label_txt_path = "%s.txt" % (label_json_path.split(".")[0].replace("/labels_json/", "/labels/"))
    img_path = label_txt_path.replace("/labels/", "/images/").replace(".txt", ".png")
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    with open(label_txt_path) as f:
        lines = f.readlines()

    for line in lines:
        cx, cy, w, h = line.strip().split(" ")[1:]
        cx = float(cx) * width
        cy = float(cy) * height
        w = float(w) * width
        h = float(h) * height
        x1 = int(cx - w / 2)
        x2 = int(x1 + w)
        y1 = int(cy - h / 2)
        y2 = int(y1 + h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    img = cv2.resize(img, (400, 300))
    cv2.imshow("img", img)
    cv2.waitKey()


def resize_imgs(img_root, img_save_root, new_height=320):
    os.makedirs(img_save_root, exist_ok=True)
    img_paths = glob("%s/*" % img_root)
    for img_path in tqdm(img_paths):
        img_path = img_path.replace("\\", "/")
        img = cv2.imread(img_path)
        if img is None:
            continue
        height, width, _ = img.shape
        ratio = new_height / height
        new_width = int(width * ratio)
        new_img = cv2.resize(img, (new_width, new_height))
        img_save_path = "%s/%s.png" % (img_save_root, os.path.basename(img_path).split(".")[0])
        cv2.imwrite(img_save_path, new_img)


def gen_train_test_txt(img_root, save_root, train_ratio=0.8):
    img_abso_paths = glob("%s/*" % (img_root))
    img_abso_paths = [x.replace("\\", "/") for x in img_abso_paths]
    train = []
    test = []
    for path in img_abso_paths:
        if random.random() <= train_ratio:
            train.append(path)
        else:
            test.append(path)

    with open("%s/train.txt" % save_root, "w") as f:
        f.write("\n".join(train))
    with open("%s/test.txt" % save_root, "w") as f:
        f.write("\n".join(test))


def img2jpg_img(img_root):
    abso_paths = glob("%s/*" % img_root)
    for abso_path in tqdm(abso_paths):
        img = cv2.imread(abso_path)
        dst_path = "%s.jpg" % (abso_path.split(".")[0])
        cv2.imwrite(dst_path, img)


LABEL_MAP = {
    "high": 0,
    "level": 1,
    "low": 2,
}

LABEL_MAP = {
    "bolt": 0,
}

# # json_label_root = "F:/dataset/bolt_piezometer/bolt/labels_json/train"
# # json_label_root = "F:/dataset/bolt_piezometer/piezometer/labels_json/train"
# # json_label_root = "F:/dataset/bolt_piezometer/piezometer_multi_cls/labels_json/train"
# # json_label_root = "F:/dataset/bolt_piezometer/piezometer_panel/labels_json/train"
dataset_name = "bolt_total"
json_label_root = "F:/dataset/bolt_piezometer/hole/labels_json"
json_label_root = "F:/dataset/bolt_piezometer/oil_level/labels_json"
json_label_root = "F:/dataset/bolt_piezometer/%s/labels_json" % dataset_name
# labelbee_json2txt(json_label_root)
# labelme_json2txt(json_label_root)

img_root = "F:/dataset/bolt_piezometer/%s/images_ori" % dataset_name
img_save_root = "F:/dataset/bolt_piezometer/%s/images" % dataset_name
# resize_imgs(img_root, img_save_root, new_height=640)
# verify(json_label_root)


img_root = "F:/dataset/bolt_piezometer/%s/images" % dataset_name
save_root = "F:/dataset/bolt_piezometer/%s" % dataset_name
gen_train_test_txt(img_root, save_root)

# img2jpg_img(r"F:\dataset\bolt_piezometer\oil_level\images")
