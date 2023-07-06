import time

import cv2
import numpy as np
from mmseg.apis import inference_segmentor, init_segmentor


def init_model_segmentator(config, ckpt, device="cuda"):
    # build the model from a config file and a checkpoint file
    model = init_segmentor(config, ckpt, device=device)
    return model


def run_segmentator(model, imgs):
    # if not isinstance(img_names, list):
    #     img_names = [img_names]
    # import pdb
    # pdb.set_trace()
    results = {}
    # each element is a string
    results["input_imgs"] = imgs
    # each element is a mask
    results["preds"] = []
    t0 = time.time()
    for input_img in imgs:
        # test a single image
        result = inference_segmentor(model, input_img)
        mask = np.zeros((*result[0].shape, 1), dtype=np.uint8)
        mask[:, :, 0] = result[0]
        mask[np.where(mask == 1)] = 255
        results["preds"].append(mask)
    t1 = time.time()
    return results


def vis_seg_results(seg_results, save_img_names):
    masks = seg_results["preds"]
    if not isinstance(save_img_names, list):
        save_img_names = [save_img_names]
    assert len(masks) == len(save_img_names)
    for i in range(len(masks)):
        mask = masks[i]
        save_img_name = save_img_names[i]
        cv2.imwrite(save_img_name, mask)


if __name__ == '__main__':
    # config = "local_configs/segformer/B0/segformer.b0.256x256.pointer.160k.py"
    # ckpt = "work_dirs/segformer.b0.256x256.pointer.160k/iter_4000.pth"

    config = "local_configs/segformer/B0/segformer.b0.256x256.bolt_line.6k.py"
    ckpt = "work_dirs/segformer.b0.256x256.bolt_line.6k/iter_5500.pth"
    device = "cuda"
    model = init_model_segmentator(config, ckpt, device)
    img_name = [
        "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/piezometer_panel_multi_cls/images/train/0_DJI_20230424102125_0157_V_000.JPG",
        "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/piezometer_panel_multi_cls/images/train/0_DJI_20230424102125_0157_V_000.JPG"
    ]
    img_name = [
        "bolt_test_00.JPG"
    ]
    run_segmentator(model, img_name)
