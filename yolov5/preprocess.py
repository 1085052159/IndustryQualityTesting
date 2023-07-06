import os
from glob import glob
import json

import cv2
from tqdm import tqdm


def json2txt(json_label_root):
    json_labels = glob("%s/*.json" % json_label_root)
    for label_path in tqdm(json_labels):
        label_path = label_path.replace("\\", "/")
        label_txt_path = "%s.txt" % (label_path.split(".")[0].replace("/labels_json/", "/labels/"))
        label_txt_root = os.path.dirname(label_txt_path)
        os.makedirs(label_txt_root, exist_ok=True)
        with open(label_path, "r") as f:
            data = json.load(f)
        img_width = int(data["width"])
        img_height = int(data["height"])
        results = data["step_1"]["result"]
        lines = []
        for result in results:
            attribute = result["attribute"]
            if len(attribute) == 0:
                continue
            x = float(result["x"])
            y = float(result["y"])
            width = float(result["width"])
            height = float(result["height"])
            cx = x + width / 2
            cy = y + height / 2
            line = "%s %s %s %s %s" % (attribute, cx / img_width, cy / img_height, width / img_width, height / img_height)
            lines.append(line)
        with open(label_txt_path, "w") as f:
            f.write("\n".join(lines))


def verify(json_label_root):
    label_json_path = glob("%s/*.json" % json_label_root)[0].replace("\\", "/")
    label_txt_path = "%s.txt" % (label_json_path.split(".")[0].replace("/labels_json/", "/labels/"))
    img_path = label_txt_path.replace("/labels/", "/images/").replace(".txt", ".JPG")
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
    cv2.imshow("img", cv2.resize(img, (400, 300)))
    cv2.waitKey()


def resize_imgs(img_root, img_save_root, new_height=320):
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
        img_save_path = "%s/%s" % (img_save_root, os.path.basename(img_path))
        cv2.imwrite(img_save_path, new_img)

# json_label_root = "F:/dataset/bolt_piezometer/bolt/labels_json/train"
# json_label_root = "F:/dataset/bolt_piezometer/piezometer/labels_json/train"
# json_label_root = "F:/dataset/bolt_piezometer/piezometer_multi_cls/labels_json/train"
# json_label_root = "F:/dataset/bolt_piezometer/piezometer_panel/labels_json/train"
json_label_root = "F:/dataset/bolt_piezometer/oil_level/labels_json/train"
json2txt(json_label_root)
verify(json_label_root)
# img_root = "F:/dataset/bolt_piezometer/bolt/images/train"
# img_root = "F:/dataset/bolt_piezometer/piezometer/images/train"
# img_save_root = "F:/dataset/bolt_piezometer/piezometer_multi_cls/images/train"
img_root = "F:/dataset/bolt_piezometer/oil_level/images/train_ori"
img_save_root = "F:/dataset/bolt_piezometer/oil_level/images/train"
resize_imgs(img_root, img_save_root, new_height=640)