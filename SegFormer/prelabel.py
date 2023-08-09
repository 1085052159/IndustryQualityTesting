import base64
import io
import json
import os
from glob import glob

import PIL.ExifTags
import PIL.Image
import PIL.ImageOps
import cv2
from tqdm import tqdm


def apply_exif_orientation(image):
    try:
        exif = image._getexif()
    except AttributeError:
        exif = None
    
    if exif is None:
        return image
    
    exif = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in exif.items()
        if k in PIL.ExifTags.TAGS
    }
    
    orientation = exif.get("Orientation", None)
    
    if orientation == 1:
        # do nothing
        return image
    elif orientation == 2:
        # left-to-right mirror
        return PIL.ImageOps.mirror(image)
    elif orientation == 3:
        # rotate 180
        return image.transpose(PIL.Image.ROTATE_180)
    elif orientation == 4:
        # top-to-bottom mirror
        return PIL.ImageOps.flip(image)
    elif orientation == 5:
        # top-to-left mirror
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_270))
    elif orientation == 6:
        # rotate 270
        return image.transpose(PIL.Image.ROTATE_270)
    elif orientation == 7:
        # top-to-right mirror
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_90))
    elif orientation == 8:
        # rotate 90
        return image.transpose(PIL.Image.ROTATE_90)
    else:
        return image


def load_image_file(filename):
    try:
        image_pil = PIL.Image.open(filename)
    except IOError:
        return
    
    # apply orientation to image according to exif
    image_pil = apply_exif_orientation(image_pil)
    
    with io.BytesIO() as f:
        image_pil.save(f, format="JPEG")
        f.seek(0)
        return f.read()


def save_label(img_path, json_path):
    raw_h, raw_w = cv2.imread(img_path).shape[:2]
    img_data = load_image_file(img_path)
    imageData = base64.b64encode(img_data).decode("utf-8")
    save_data = {
        "version": "1.0.0",
        "flags": {},
        "shapes": [],
        "imagePath": img_path,
        "imageData": imageData,
        "imageHeight": raw_h,
        "imageWidth": raw_w
    }
    
    with open(json_path, 'w') as f:
        json.dump(save_data, f)
    return True


def save_labels(img_root, json_root):
    img_paths = glob("%s/*" % img_root)
    for img_path in tqdm(img_paths):
        json_path = "%s/%s.json" % (json_root, os.path.basename(img_path).split(".")[0])
        if os.path.exists(json_path):
            continue
        save_label(img_path, json_path)


img_root = "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/bolt_single_total/images"
json_root = "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/bolt_single_total/labels_json"
save_labels(img_root, json_root)
