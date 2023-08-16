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
sys.path.append(os.getcwd() + "/SegFormer")
import math
import shutil
import time
import cv2
import numpy as np

from tqdm import tqdm
from glob import glob
from PIL import ImageFont, ImageDraw, Image

from yolov5.detect_infer import init_model_detector, run_detector, vis_det_results
from SegFormer.seg_infer import init_model_segmentator, run_segmentator, vis_seg_results


COLORS = [
    (0, 255, 0),
    (0, 0, 255),
    (0, 0, 255),
    (0, 0, 255),
    (0, 255, 255),
    (0, 0, 255),
]


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


def bolt_post_process(preds, img, new_height=320):
    """
    1. 使用训练图预测，得到预测的归一化bbox
    2. 在原图上crop
    3. 裁剪后的图，resize到新尺寸：pred_norm_h/w * ori_h/ori_w * ratio
    :param preds: [[numpy_arr1, ...], [numpy_arr1, numpy_arr2, ...]]
    :param img_names: [str1, str2]
    :param save_path:
    :param new_height:
    :return
    cropped_imgs: [[]], a list, each element is a list, each element is an image array
    """
    # import pdb
    # pdb.set_trace()
    cropped_img = []
    norm_bboxes, bboxes = [], []
    height, width, _ = img.shape
    for idx, pred in enumerate(preds):
        cls_id, cx, cy, w, h, _ = pred
        bbox = norm_cxy2xxyy([cx, cy, w, h])
        x1, y1, x2, y2 = norm2image_level(bbox, height, width)
        crop_img = img[y1: y2 + 1, x1: x2 + 1, :]
        ori_h = h * height
        ratio = new_height / ori_h
        dst_w = int(w * width * ratio)
        if dst_w > 3 * new_height or dst_w < 1 / 3 * new_height:
            continue
        crop_img = cv2.resize(crop_img, (dst_w, int(new_height)))
        cropped_img.append(crop_img)
        norm_bboxes.append([str(x) for x in [cls_id, *bbox, _]])
        bboxes.append([str(x) for x in [cls_id, x1, y1, x2, y2, _]])
    return cropped_img, norm_bboxes, bboxes


def get_one_mask_bboxes(mask):
    if len(mask.shape) == 3:
        mask_ori = mask[:, :, 0]
    else:
        mask_ori = mask.copy()
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 20:
            cv2.fillPoly(mask_ori, [contour], 0, 1)
            continue
        rotate_rect = cv2.minAreaRect(contour)
        bboxes.append(rotate_rect)
    
    # import pdb
    # pdb.set_trace()
    ys, xs = np.nonzero(mask_ori)
    if len(ys) > 0 and len(xs) > 0:
        contour = np.array([[xs[i], ys[i]] for i in range(len(ys))])
        rotate_rect = cv2.minAreaRect(contour)
        bboxes.append(rotate_rect)
    return bboxes


def get_loosen_state(bboxes, thresh_angle=5, thresh_dist=5):
    """
    1. 当只有一个bbox时，螺丝不松
    2. 当有两个bbox时，若bbox的中心点角度超过阈值或两个bbox的间隔超过阈值，螺丝松动
    3. 当超过两个bbox时，
    :param bboxes: list, each element is [cx, cy, w, h], not normalized
    :param thresh_angle: 角度阈值，当超过该阈值，则认为螺丝松动
    :param thresh_dist: 距离阈值，当长或宽的百分比超过该阈值，则认为螺丝松动
    :return:
        0: 未松动
        1: 角度偏离超过阈值，螺丝松动
        2: 水平偏离超过阈值，螺丝松动
        3: 垂直偏离超过阈值，螺丝松动
        4: 无标记线，无法判断
        5: 标记线不规范，无法判断
    """
    if len(bboxes) == 0:
        state = 4
    elif len(bboxes) == 2:
        state = 0
    else:
        state = 0
        out_rect = bboxes.pop()
        out_angle = out_rect[-1]
        out_angle = out_angle if out_angle <= 45 else 90 - out_angle
        # print("out_rect: ", out_rect, out_angle)
        for idx, rect in enumerate(bboxes):
            angle = rect[-1]
            angle = angle if angle <= 45 else 90 - angle
            # print("rect: ", rect, angle)
            if angle == out_angle:
                if abs(out_rect[1][0] - rect[1][0]) <= 2 or abs(out_rect[1][1] - rect[1][1]) <= 2:
                    state = 0
                    break
            else:
                if not abs(out_angle - angle) <= thresh_angle:
                    state = 1
    return state


def masks_post_process(results_bolt_marker, thresh_angle=5, thresh_dist=5):
    bolt_marker_masks = results_bolt_marker["preds"]
    states = []
    for i in range(len(bolt_marker_masks)):
        # list, each element is [cx, cy, w, h]
        bbox = get_one_mask_bboxes(bolt_marker_masks[i])
        loosen_state = get_loosen_state(bbox, thresh_angle, thresh_dist)
        states.append(loosen_state)
    return states


def recog_one_img_bolt_state(source_ori,
                             model_det_bolt, model_seg_bolt_marker,
                             thresh_angle=5, thresh_dist=5,
                             save_path="tmp_results/bolt_loosen",
                             show_temp=False, device="cuda:0"):
    """
    :param source_ori: 输入图片的绝对路径
    :param model_det_bolt: 螺丝检测模型
    :param model_seg_bolt_marker: 螺丝标记线分割模型
    :param save_path: 结果保存路径，默认保存在工程目录下的tmp_results/piezometer_reading
    :param show_temp: 是否保存算法临时结果，用于debug，True表示保存，False表示不保存
    :param device: 推理设备，默认gpu
    :return:
    bolt_loosen_result: json格式
    {
        "input_info": {
            "height": 整数，图像高度
            "width": 整数，图像宽度
            "img_path": 字符串，图像路径
            # "img_numpy": numpy数组，维度为(h, w, 3)，图像矩阵
        }
        "output_info": {
            "norm_bbox": [[cls_id, norm_cx, norm_cy, norm_w, norm_h, conf],
                          [cls_id, norm_cx, norm_cy, norm_w, norm_h, conf], ...]
                         螺丝在图像中的相对位置，二维列表，每个元素是一个长度为6的列表，依次表示:
                         类别id, 归一化后矩形框中心点x坐标, 归一化后矩形框中心点y坐标, 归一化后矩形框宽, 归一化后矩形框高, 置信度
            "bbox": [[cls_id, cx, cy, w, h, conf],
                     [cls_id, cx, cy, w, h, conf], ...]
                    螺丝在图像中的绝对位置，二维列表，每个元素是一个长度为6的列表，依次表示:
                    类别id, 矩形框中心点x坐标, 矩形框中心点y坐标, 矩形框宽, 矩形框高, 置信度
            "state": [螺丝1状态, 螺丝2状态, ...]，
                     螺丝识别结果，一维列表，每个元素表示螺丝读数结果，读数结果为整数，有6种不同状态，分别是0-5
            "state_desc": {
                "0": "未松动",
                "1": "角度偏离超过阈值，螺丝松动",
                "2": "水平偏离超过阈值，螺丝松动",
                "3": "垂直偏离超过阈值，螺丝松动",
                "4": "无标记线，无法判断",
                "5": "标记线不规范，无法判断",
            }
        }
    }
    """
    bolt_loosen_result = {}
    bolt_loosen_result["input_info"] = {}
    bolt_loosen_result["output_info"] = {}
    bolt_loosen_result["output_info"]["state_desc"] = {}
    state_list = ["未松动", "角度偏离超过阈值，螺丝松动", "水平偏离超过阈值，螺丝松动",
                  "垂直偏离超过阈值，螺丝松动", "无标记线，无法判断", "标记线不规范，无法判断"]
    for i in range(len(state_list)):
        bolt_loosen_result["output_info"]["state_desc"][str(i)] = state_list[i]
    
    img0, filename = read_img(source_ori)
    ori_h, ori_w, _ = img0.shape
    imgsz = 640
    # imgsz = 416
    img = resize_img(img0, ratio=(imgsz / ori_h))
    
    bolt_loosen_result["input_info"]["img_path"] = source_ori
    bolt_loosen_result["input_info"]["height"] = ori_h
    bolt_loosen_result["input_info"]["width"] = ori_w
    
    if show_temp:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
            print("%s exist! Delete!" % save_path)
    
    ##########################################
    # get bolt results
    ##########################################
    det_t0 = time.time()
    bolt_preds = run_detector(model_det_bolt,  # model.pt path(s)
                              source=[img],  # file/dir/URL/glob, 0 for webcam
                              imgsz=imgsz,  # inference size (pixels)
                              conf_thres=0.25,  # confidence threshold
                              iou_thres=0.2,  # NMS IOU threshold
                              max_det=1000,  # maximum detections per image
                              device=device,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                              )
    det_t1 = time.time()
    det_time = det_t1 - det_t0
    
    if show_temp:
        bolt_save_path = "%s/bolts" % save_path
        os.makedirs(bolt_save_path, exist_ok=True)
        save_names = ["%s/%s.png" % (bolt_save_path, filename)]
        vis_det_results(bolt_preds, save_names, line_thickness=3)
    
    # 未检测到螺丝时，直接返回
    if len(bolt_preds["preds"]) == 0:
        bolt_loosen_result["output_info"]["norm_bbox"] = []
        bolt_loosen_result["output_info"]["bbox"] = []
        bolt_loosen_result["output_info"]["state"] = []
        return bolt_loosen_result
    
    ##########################################
    # 根据螺丝检测结果获取裁剪后的螺丝图片和名字
    ##########################################
    det_post_t0 = time.time()
    single_bolt_imgs, norm_bboxes, bboxes = bolt_post_process(bolt_preds["preds"][0], img0, new_height=256)
    bolt_loosen_result["output_info"]["norm_bbox"] = norm_bboxes
    bolt_loosen_result["output_info"]["bbox"] = bboxes
    det_post_t1 = time.time()
    det_post_time = det_post_t1 - det_post_t0
    
    if show_temp:
        single_bolt_save_root = "%s/bolts_crop" % save_path
        single_bolt_img_names = ["%s/%s_%s.png" % (single_bolt_save_root, filename,
                                                   str(idx).zfill(2)) for idx in range(len(single_bolt_imgs))]
        os.makedirs(single_bolt_save_root, exist_ok=True)
        write_img(single_bolt_img_names, single_bolt_imgs)
    
    ##########################################
    # get bolt marker mask according to the bolt seg results
    ##########################################
    seg_t0 = time.time()
    results_bolt_marker = run_segmentator(model_seg_bolt_marker, single_bolt_imgs)
    seg_t1 = time.time()
    seg_time = seg_t1 - seg_t0
    # print("run_seg_time: %.3f" % (seg_time))
    
    if show_temp:
        mark_line_save_root = "%s/bolts_marker_line" % save_path
        os.makedirs(mark_line_save_root, exist_ok=True)
        mask_save_paths = ["%s/%s" % (mark_line_save_root, os.path.basename(name)) for name in single_bolt_img_names]
        vis_seg_results(results_bolt_marker, mask_save_paths)
    
    seg_post_t0 = time.time()
    states = masks_post_process(results_bolt_marker, thresh_angle, thresh_dist)
    bolt_loosen_result["output_info"]["state"] = states
    seg_post_t1 = time.time()
    seg_post_time = seg_post_t1 - seg_post_t0
    # print("run_seg_post_time: %.3f" % (seg_post_time))
    
    total_time = det_time + det_post_time + seg_time + seg_post_time
    print("total_time: %.3f; run_det_time: %.3f; run_det_post_time: %.3f; run_seg_time: %.3f; "
          "run_seg_post_time: %.3f" % (
              total_time, det_time, det_post_time, seg_time, seg_post_time
          ))
    if show_temp:
        recog_save_root = "%s/result" % save_path
        os.makedirs(recog_save_root, exist_ok=True)
        assert len(single_bolt_imgs) == len(states)
        for i in range(len(states)):
            img = single_bolt_imgs[i]
            # import pdb
            # pdb.set_trace()
            img = cv2.putText(img, str(states[i]), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[states[i]], 1)
            img_save_path = "%s/%s" % (recog_save_root, os.path.basename(single_bolt_img_names[i]))
            cv2.imwrite(img_save_path, img)
    return bolt_loosen_result


def save_bolt_state_result(bolt_loosen_result, save_path):
    img_path = bolt_loosen_result["input_info"]["img_path"]
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    state_desc = bolt_loosen_result["output_info"]["state_desc"]
    # 在图像左上角表明类别注释
    font_path = "simsun.ttc"
    if h > 1200:
        font_size = 100
        stroke_width = 2
        font_scale = 6
        thickness = 6
    elif h > 800:
        font_size = 30
        stroke_width = 1
        font_scale = 2
        thickness = 3
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    for key, value in state_desc.items():
        line = "%s: %s" % (key, value)
        draw.text((30, 30 + int(key) * font_size), line, COLORS[int(key)], font=font, stroke_width=stroke_width)
    img = np.array(img_pil)
    
    bboxes = bolt_loosen_result["output_info"]["bbox"]
    states = bolt_loosen_result["output_info"]["state"]
    # 每个图片的螺丝位置
    for i in range(len(bboxes)):
        # 每个螺丝的位置信息
        cls_id, x1, y1, x2, y2, conf = bboxes[i]
        x1, y1, x2, y2 = [int(x) for x in [x1, y1, x2, y2]]
        
        state = states[i]
        # print(img_path, x1, y1, x2, y2)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[state], 6)
        img = cv2.putText(img, str(state), (x1 - 10, y1 - font_scale * 10),
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLORS[state], thickness)
    os.makedirs(save_path, exist_ok=True)
    base_name = os.path.basename(img_path)
    img_save_path = "%s/%s" % (save_path, base_name)
    cv2.imwrite(img_save_path, img)


def recog_img_bolt_state(abso_img_paths, save_path,
                         thresh_angle=5, thresh_dist=5,
                         device="cuda", show_temp=False):
    """
    :param abso_img_paths: 输入图片的绝对路径，字符串列表
    :param save_path: 结果保存路径
    :param kernel_size: 对分割结果进行腐蚀操作的核大小，为1时，不进行腐蚀操作
    :param show_temp: 是否保存算法临时结果，用于debug，True表示保存，False表示不保存
    :return:
    results: 列表，每个元素表示对应图片的识别结果，结果格式见recog_one_img_bolt_state方法的返回结果
    """
    ckpt_bolt_det = "./ckpts/bolt_det.pt"
    # init bolt detector
    model_det_bolt = init_model_detector(ckpt_bolt_det, device)
    
    # init bolt marker segmentator
    config_bolt_marker_seg = "./ckpts/bolt_line_seg_config.py"
    ckpt_bolt_marker_seg = "./ckpts/bolt_line_seg_backup.pth"
    model_seg_bolt_marker = init_model_segmentator(config_bolt_marker_seg, ckpt_bolt_marker_seg, device)
    
    results = []
    for source_ori in tqdm(abso_img_paths):
        save_path_ = "%s/%s" % (save_path, os.path.basename(source_ori).split(".")[0])
        if not os.path.exists(source_ori):
            print("%s not exist" % source_ori)
            continue
        bolt_state_result = recog_one_img_bolt_state(source_ori,
                                                     model_det_bolt, model_seg_bolt_marker,
                                                     thresh_angle, thresh_dist,
                                                     "%s/alg_out" % save_path_, show_temp, device)
        results.append(bolt_state_result)
        if show_temp:
            # 保存识别结果
            save_bolt_state_result(bolt_state_result, "%s/result" % save_path_)
    return results


def main():
    img_root = "./test_imgs/bolt4"
    img_root = "./vid_frames1/DJI_20230728095908_0005_V"
    abso_img_paths = sorted(glob("%s/*" % img_root))
    # abso_img_paths = [
    #     "DJI_20230721095605_0018_V.JPG",
    # ]
    # abso_img_paths = ["%s/%s" % (img_root, x) for x in abso_img_paths]
    save_path = "tmp_results/bolt_loosen4"
    save_path = "tmp_results/DJI_20230728095908_0005_V"
    thresh_angle = 8
    thresh_dist = 8
    device = "cuda"
    show_temp = True
    # show_temp = False
    results = recog_img_bolt_state(abso_img_paths, save_path,
                                   thresh_angle, thresh_dist,
                                   device, show_temp)
    print(results)


if __name__ == '__main__':
    main()
