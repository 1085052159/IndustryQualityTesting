import os
import shutil
import uuid
from glob import glob

import cv2
import numpy as np
import torch
from torch import nn
import torchvision
from tqdm import tqdm


def vid2frames(vid_path, interval=1):
    reader = cv2.VideoCapture(vid_path)
    fps = reader.get(cv2.CAP_PROP_FPS)
    count = reader.get(cv2.CAP_PROP_FRAME_COUNT)
    print("FPS: %s, total count: %s" % (fps, count))
    frames = []
    idx = 0
    while True:
        ret, frame = reader.read()
        if not ret:
            print("vid_path: %s; frames: %s" % (vid_path, len(frames)))
            break
        if idx % interval == 0:
            frames.append(frame)
        idx += 1
    return frames


def gen_name(count):
    unique_id = uuid.uuid1()
    hex_id = unique_id.hex
    hex_id = hex_id.replace("-", "")
    truncated_id = hex_id[:8]
    return "%06d_%s" % (count, truncated_id)


def write_frames(frames, save_path):
    os.makedirs(save_path, exist_ok=True)
    for i in tqdm(range(0, len(frames))):
        cv2.imwrite("%s/%s.png" % (save_path, gen_name(i)), frames[i])


def batch_rename():
    base_root = "./vid_frames1"
    vid_names = os.listdir(base_root)
    for vid_name in tqdm(vid_names):
        img_paths = glob("%s/%s/*" % (base_root, vid_name))
        for idx, img_path in enumerate(img_paths):
            os.rename(img_path, "%s/%s/%s.png" % (base_root, vid_name, gen_name(idx)))


def filter_(save_path):
    os.makedirs(save_path, exist_ok=True)
    base_root = "./yolov5/runs/detect/bolt_640_0.25_0.2/bolt/total"
    img_paths = glob("%s/*.jpg" % base_root)
    count = 0
    del_img = DelSimImgByVGG()
    for idx, img_path in tqdm(enumerate(img_paths)):
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        ratio = h / w
        if h < 50 or w < 50 or ratio > 2.5 or ratio < 1 / 2.5:
            count += 1
            os.remove(img_path)
        delete = del_img.del_sim_img_by_vgg(img / 255.0, thr=0.01)
        if not delete:
            shutil.copy(img_path, "%s/%s" % (save_path, os.path.basename(img_path)))
        if idx != 0 and idx % 30 == 0:
            del_img.saved_loss.clear()
    
    print("count: ", count, len(img_paths))


class DelSimImgByVGG:
    def __init__(self, device="cuda"):
        vgg = torchvision.models.vgg19(pretrained=True).features
        vgg.to(device)
        vgg.eval()
        self.vgg = vgg
        self.device = device
        self.saved_loss = []
    
    def del_sim_img_by_vgg(self, img, thr=0.01):
        if len(img.shape) == 3:
            img_ = img.transpose(2, 0, 1)
            img_ = np.array(img_, dtype=np.float32)
            img_ = torch.from_numpy(img_).unsqueeze(0)
            img_ = img_.to(self.device)
        else:
            img_ = img
        
        loss = self.vgg(img_).mean()
        loss = loss.detach().cpu().numpy()
        # print("loss: ", loss)
        delete = False
        if len(self.saved_loss) == 0:
            self.saved_loss.append(loss)
            return delete
        for loss_ in self.saved_loss.copy():
            if abs(loss_ - loss) <= thr:
                delete = True
                break
            else:
                self.saved_loss.append(loss)
        return delete


# vid_path = "video02.mp4"
vid_path = "/media/ubuntu/dataset_nvme/BaiduNetdiskDownload/20230728/DJI_20230728094500_0003_V.MP4"
save_path = "vid_frames1/%s" % (os.path.basename(vid_path).split(".")[0])
# frames = vid2frames(vid_path, 10)
# write_frames(frames, save_path)
# batch_rename()
save_path = "/media/ubuntu/win_software/PycharmWorkspaces/IndustryQualityTesting/yolov5/runs/detect/bolt_640_0.25_0.2/bolt/total_"
filter_(save_path)
