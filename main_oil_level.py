"""
1. yolov5检测螺丝
2. SegFormer分割螺丝标记线
3. 松紧判断
    a. 仅有一个mask时，不会松
    b. 有两个mask时，从角度，水平偏离、竖直偏离三方面判断，当任意一者超过阈值，判定为松
    c. 当没有mask或有多个mask，判定为标记线不规范，此时默认为松
"""
import sys
import os

sys.path.append(os.getcwd() + "/yolov5")
import numpy as np
import shutil
import time
import cv2

from tqdm import tqdm
from glob import glob
from PIL import ImageFont, ImageDraw, Image

from yolov5.detect_infer import init_model_detector, run_detector, vis_det_results


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


def write_img(img_names, imgs):
    if isinstance(img_names, list):
        assert len(img_names) == len(imgs)
    else:
        img_names = [img_names]
        imgs = [imgs]
    
    # import pdb
    # pdb.set_trace()
    # 不同图像
    for i in range(len(img_names)):
        img = imgs[i]
        img_name = img_names[i]
        # 来自同一图像的裁剪图像
        if not isinstance(img, list):
            img = [img]
        if not isinstance(img_name, list):
            img_name = [img_name]
        for j in range(len(img)):
            cv2.imwrite(img_name[j], img[j])


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
    return [int(x) for x in bbox]


def find_max_idx(pred):
    max_conf = pred[0][-1]
    max_idx = 0
    for i in range(1, len(pred)):
        if pred[i][-1] >= max_conf:
            max_conf = pred[i][-1]
            max_idx = i
    return max_idx


def oil_level_post_process(preds, img):
    """
    :param preds: [[numpy_arr1, ...], [numpy_arr1, numpy_arr2, ...]]
    :param img_names: [str1, str2]
    :param save_path:
    :param new_height:
    :return
    cropped_imgs: [[]], a list, each element is a list, each element is an image array
    """
    # import pdb
    # pdb.set_trace()
    norm_bboxes, bboxes = [], []
    height, width, _ = img.shape
    
    high = preds[np.where(preds[:, 0] == 0)[0], :]  # 高液位阈值
    level = preds[np.where(preds[:, 0] == 1)[0], :]  # 油液
    low = preds[np.where(preds[:, 0] == 2)[0], :]  # 低液位阈值
    
    """
    同一个位置会预测多个类别，但是错误类别的置信度低于正确类别
    """
    if len(high) > 1:
        max_idx = find_max_idx(high)
        high = [high[max_idx]]
    if len(level) > 1:
        max_idx = find_max_idx(level)
        level = [level[max_idx]]
    if len(low) > 1:
        max_idx = find_max_idx(low)
        low = [low[max_idx]]
    # import pdb
    # pdb.set_trace()
    preds = []
    if len(high) > 0:
        preds.append(high[0])
    if len(level) > 0:
        preds.append(level[0])
    if len(low) > 0:
        preds.append(low[0])
    
    for idx, pred in enumerate(preds):
        cls_id, cx, cy, w, h, _ = pred
        bbox = norm_cxy2xxyy([cx, cy, w, h])
        x1, y1, x2, y2 = norm2image_level(bbox, height, width)
        norm_bboxes.append([str(x) for x in [int(cls_id), *bbox, _]])
        bboxes.append([str(x) for x in [int(cls_id), x1, y1, x2, y2, _]])
    return norm_bboxes, bboxes


def get_oil_level_state(bboxes, thresh_high=0.05, thresh_low=0.01):
    """
    :param bboxes: list, each element is [cls_id, x1, y1, x2, y2, conf], not normalized
    :param thresh_high: 高位阈值，当超过该阈值，则认为高位预警状态
    :param thresh_low: 低位阈值，当超过该阈值，则认为低位预警状态
    :return:
        0: "正常",
        1: "超过高位阈值",
        2: "低于低位阈值",
        3: "其它",
        4: "疑似高位"
        5: "疑似低位"
    """
    # import pdb
    # pdb.set_trace()
    high_cy, level_cy, low_cy = 0, 0, 0
    
    for bbox in bboxes:
        bbox = [float(x) for x in bbox]
        if bbox[0] == 0:
            high_cy = bbox[2] + (bbox[4] - bbox[2]) / 2
        if bbox[0] == 1:
            level_cy = bbox[2] + (bbox[4] - bbox[2]) / 2
        if bbox[0] == 2:
            low_cy = bbox[2] + (bbox[4] - bbox[2]) / 2
    if level_cy == 0 or (high_cy == 0 and low_cy == 0):
        return 3
    # import pdb
    # pdb.set_trace()
    dist_high_low = low_cy - high_cy
    if level_cy <= high_cy - thresh_high * dist_high_low:
        state = 1  # 超过高位
    elif level_cy <= high_cy + thresh_high * dist_high_low:
        state = 4  # 疑似高位
    elif level_cy <= low_cy - thresh_low * dist_high_low:
        state = 0  # 正常
    elif level_cy <= low_cy + thresh_low * dist_high_low:
        state = 5  # 疑似低位
    else:
        state = 2  # 低于低位
    return state


def recog_one_img_oil_level(source_ori,
                            model_det_oil_level,
                            save_path="tmp_results/oil_level",
                            show_temp=False, device="cuda:0"):
    """
    :param source_ori: 输入图片的绝对路径
    :param model_det_oil_level: 油液检测模型
    :param save_path: 结果保存路径，默认保存在工程目录下的tmp_results/oil_level
    :param show_temp: 是否保存算法临时结果，用于debug，True表示保存，False表示不保存
    :param device: 推理设备，默认gpu
    :return:
    oil_level_result: json格式
    {
        "input_info": {
            "height": 整数，图像高度
            "width": 整数，图像宽度
            "img_path": 字符串，图像路径
        }
        "output_info": {
            "norm_bbox": [[cls_id, norm_cx, norm_cy, norm_w, norm_h, conf],
                          [cls_id, norm_cx, norm_cy, norm_w, norm_h, conf], ...]
                         高阈值/液面/低阈值在图像中的相对位置，二维列表，每个元素是一个长度为6的列表，依次表示:
                         类别id, 归一化后矩形框中心点x坐标, 归一化后矩形框中心点y坐标, 归一化后矩形框宽, 归一化后矩形框高, 置信度
                         类别id=0时, 油管; 类别id=1时, 油液
            "bbox": [[cls_id, cx, cy, w, h, conf],
                     [cls_id, cx, cy, w, h, conf], ...]
                    高阈值/液面/低阈值在图像中的绝对位置，二维列表，每个元素是一个长度为6的列表，依次表示:
                    类别id, 矩形框中心点x坐标, 矩形框中心点y坐标, 矩形框宽, 矩形框高, 置信度
                    类别id=0时, 油管; 类别id=1时, 油液
            "state": [油液状态]，
                     油液识别结果，一维列表，每个元素表示油液状态，有4种不同状态，分别是0-3
            "state_desc": {
                "0": "正常",
                "1": "超过高位阈值",
                "2": "低于低位阈值",
                "3": "其它",
                "4": "疑似高位"
                "5": "疑似低位"
            }
        }
    }
    """
    oil_level_result = {}
    oil_level_result["input_info"] = {}
    oil_level_result["output_info"] = {}
    
    oil_level_result["output_info"]["state_desc"] = {}
    state_list = ["正常", "超过高位阈值", "低于低位阈值", "其它", "疑似高位", "疑似低位"]
    for i in range(len(state_list)):
        oil_level_result["output_info"]["state_desc"][str(i)] = state_list[i]
    
    img0, filename = read_img(source_ori)
    ori_h, ori_w, _ = img0.shape
    # imgsz = 320
    imgsz = 640
    img = resize_img(img0, ratio=(imgsz / ori_h))
    
    oil_level_result["input_info"]["img_path"] = source_ori
    oil_level_result["input_info"]["height"] = ori_h
    oil_level_result["input_info"]["width"] = ori_w
    
    if show_temp:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
            print("%s exist! Delete!" % save_path)
    
    det_t0 = time.time()
    oil_level_preds = run_detector(model_det_oil_level,  # model.pt path(s)
                                   source=[img],  # file/dir/URL/glob, 0 for webcam
                                   imgsz=imgsz,  # inference size (pixels)
                                   conf_thres=0.25,  # confidence threshold
                                   iou_thres=0.45,  # NMS IOU threshold
                                   max_det=1000,  # maximum detections per image
                                   device=device,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                                   )
    det_t1 = time.time()
    det_time = det_t1 - det_t0
    # import pdb
    # pdb.set_trace()
    if show_temp:
        oil_level_save_path = "%s/oil_levels" % save_path
        os.makedirs(oil_level_save_path, exist_ok=True)
        save_names = ["%s/%s.png" % (oil_level_save_path, filename)]
        vis_det_results(oil_level_preds, save_names, line_thickness=1)
    
    # 未检测到油液，直接返回
    if len(oil_level_preds["preds"]) == 0:
        oil_level_result["output_info"]["norm_bbox"] = []
        oil_level_result["output_info"]["bbox"] = []
        oil_level_result["output_info"]["level"] = []
        return oil_level_result
    
    det_post_t0 = time.time()
    norm_bboxes, bboxes = oil_level_post_process(oil_level_preds["preds"][0], img0)
    oil_level_result["output_info"]["norm_bbox"] = norm_bboxes
    oil_level_result["output_info"]["bbox"] = bboxes
    det_post_t1 = time.time()
    det_post_time = det_post_t1 - det_post_t0
    
    ##########################################
    # 根据油液检测结果获取当前油液状态
    ##########################################
    calc_t0 = time.time()
    state = get_oil_level_state(bboxes)
    oil_level_result["output_info"]["state"] = [str(state)]
    clac_t1 = time.time()
    calc_time = clac_t1 - calc_t0
    
    total_time = det_time + det_post_time + calc_time
    print("total_time: %.3f; run_det_time: %.3f; run_det_post_time: %.3f; run_clac_time: %.3f" % (
        total_time, det_time, det_post_time, calc_time))
    if show_temp:
        recog_save_root = "%s/result" % save_path
        os.makedirs(recog_save_root, exist_ok=True)
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
        img0_ = img0.copy()
        for i in range(len(bboxes)):
            cls_id, x1, y1, x2, y2, _ = bboxes[i]
            cls_id, x1, y1, x2, y2 = [int(x) for x in [cls_id, x1, y1, x2, y2]]
            img0_ = cv2.rectangle(img0_, (x1, y1), (x2, y2), colors[int(cls_id)], 3)
        img0_ = cv2.putText(img0_, str(state), (100, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 255), 10)
        img_save_path = "%s/%s" % (recog_save_root, os.path.basename(source_ori))
        cv2.imwrite(img_save_path, img0_)
    return oil_level_result


def save_oil_level_result(oil_level_result, save_path):
    img_path = oil_level_result["input_info"]["img_path"]
    img = cv2.imread(img_path)
    states = oil_level_result["output_info"]["state"]
    
    state_desc = oil_level_result["output_info"]["state_desc"]
    # 在图像左上角表明类别注释
    font_path = "simsun.ttc"
    font_size = 100
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    line = "%s: %s" % (states[0], state_desc[states[0]])
    draw.text((10, 10), line, (0, 0, 255), font=font, stroke_width=2)
    img = np.array(img_pil)
    
    os.makedirs(save_path, exist_ok=True)
    base_name = os.path.basename(img_path)
    img_save_path = "%s/%s" % (save_path, base_name)
    cv2.imwrite(img_save_path, img)


def recog_img_oil_level(abso_img_paths, save_path,
                        device="cuda", show_temp=False):
    """
    :param abso_img_paths: 输入图片的绝对路径，字符串列表
    :param save_path: 结果保存路径
    :param show_temp: 是否保存算法临时结果，用于debug，True表示保存，False表示不保存
    :return:
    results: 列表，每个元素表示对应图片的识别结果，结果格式见recog_one_img_oil_level方法的返回结果
    """
    ckpt_oil_level_det = "./ckpts/oil_level_det.pt"
    # init oil_level detector
    model_det_oil_level = init_model_detector(ckpt_oil_level_det, device)
    
    results = []
    for source_ori in tqdm(abso_img_paths):
        save_path_ = "%s/%s" % (save_path, os.path.basename(source_ori).split(".")[0])
        if not os.path.exists(source_ori):
            print("%s not exist" % source_ori)
            continue
        oil_level_result = recog_one_img_oil_level(source_ori, model_det_oil_level,
                                                   "%s/alg_out" % save_path_, show_temp, device)
        results.append(oil_level_result)
        if show_temp:
            # 保存识别结果
            save_oil_level_result(oil_level_result, "%s/result" % save_path_)
    return results


def main():
    img_root = "./test_imgs/oil_level"
    img_root = "F:/dataset/bolt_piezometer/oil_level/images_ori"
    img_root = "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/oil_level/images_ori"
    abso_img_paths = sorted(glob("%s/*" % img_root))
    abso_img_paths = [
        "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/oil_level/images_ori/IMG_20230711_102447.jpg",
        "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/oil_level/images_ori/IMG_20230711_102451.jpg",
        "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/oil_level/images_ori/IMG_20230711_102925.jpg",
        "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/oil_level/images_ori/IMG_20230711_103317_1.jpg",
        "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/oil_level/images_ori/IMG_20230711_103355.jpg",
        "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/oil_level/images_ori/IMG_20230711_103929.jpg",
        "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/oil_level/images_ori/IMG_20230711_104415.jpg",
        "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/oil_level/images_ori/IMG_20230711_103307.jpg",
        "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/oil_level/images_ori/IMG_20230711_102909.jpg",
        "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/oil_level/images_ori/IMG_20230711_102502_1.jpg",
        "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/oil_level/images_ori/IMG_20230711_101227.jpg",
        "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/oil_level/images_ori/IMG_20230711_095713.jpg",
        "/media/ubuntu/dataset_nvme/dataset/bolt_piezometer/oil_level/images_ori/IMG_20230711_095205.jpg",
    ]
    save_path = "tmp_results/oil_level1"
    device = "cuda"
    show_temp = True
    results = recog_img_oil_level(abso_img_paths, save_path,
                                  device, show_temp)
    print(results)


if __name__ == '__main__':
    main()
