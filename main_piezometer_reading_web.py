"""
获取表计读数
1. yolov5检测压力表
    a. 检测结果会有多个，每个
2. yolov5检测压力表上mark(数字)
3. 指针分割网络获取指针mask
4. 获取半指针中点
    a. mask进行erode去噪(可选)
    b. 获取mask的最小外接矩形
    c. 计算其1/4和3/4处坐标，将mask求和，使其只保留h或w，然后依据坐标，获取值，值小的就是针尖
5. 根据欧式距离找到mask最近的两个表盘mark，获取两个mark的中点
    a. 只有单圈刻度
    b. 有两圈刻度，依据mark id，分为两圈，然后计算
6. 两个mark中间和半指针中点组成三个向量，计算夹角占比，然后乘刻度 + 起始刻度即可
"""
import sys
import os

sys.path.append(os.getcwd() + "/yolov5")
sys.path.append(os.getcwd() + "/SegFormer")
import shutil
import time
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

from yolov5.detect_infer import init_model_detector, run_detector, vis_det_results
from SegFormer.seg_infer import init_model_segmentator, run_segmentator, vis_seg_results
from flask import Flask, request, make_response, jsonify

app = Flask(__name__)
app.config.from_object(__name__)
app.config["JSON_AS_ASCII"] = False

MARK_LABEL_MAP = {
    "0": 0,
    "1": 0.1,
    "2": 0.2,
    "3": 0.3,
    "4": 0.4,
    "5": 0.5,
    "6": 0.6,
    "7": 0.8,
    "8": 1,
    "9": 1.2,
    "10": 1.6,
    "11": 10,
    "12": 20,
    "13": 30,
    "14": 40,
    "15": 50,
    "16": 60,
    "17": 80,
    "18": -20,
    "19": -40,
    "20": "pc"  # panel_center
}
MAX_INNER_MARK_CLS = 19
MIN_OUTER_MARK_CLS = 21
CLS_PANEL_CENTER = 20


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


def crop_one_img(preds, img, dst_h=320):
    height, width, _ = img.shape
    
    cropped_img = []
    for idx, pred in enumerate(preds):
        cls_id, cx, cy, w, h = pred
        bbox = norm_cxy2xxyy([cx, cy, w, h])
        x1, y1, x2, y2 = norm2image_level(bbox, height, width)
        crop_img = img[y1: y2 + 1, x1: x2 + 1, :]
        ori_h = h * height
        ratio = dst_h / ori_h
        dst_w = int(w * width * ratio)
        if dst_w < 200 or dst_w > 3 * dst_h:
            continue
        crop_img = cv2.resize(crop_img, (dst_w, int(dst_h)))
        cropped_img.append(crop_img)
    return cropped_img


def piezometer_post_process(preds, img, new_height=320):
    """
    1. 使用训练图预测，得到预测的归一化bbox
    2. 在原图上crop
    3. 裁剪后的图，resize到新尺寸：pred_norm_h/w * ori_h/ori_w * ratio
    :param preds: [[numpy_arr1, ...], [numpy_arr1, numpy_arr2, ...]]
    :param filenames: [str1, str2]
    :param new_height:
    :return
    cropped_imgs: [[]], a list, each element is a list, each element is an image array
    rest_img_names: [[]], a list, each element is a list, each element is a str
    """
    cropped_img = []
    norm_bboxes, bboxes = [], []
    height, width, _ = img.shape
    for idx, pred in enumerate(preds):
        cls_id, cx, cy, w, h, _ = pred
        bbox = norm_cxy2xxyy([cx, cy, w, h], scale_coef=1.1)
        x1, y1, x2, y2 = norm2image_level(bbox, height, width)
        crop_img = img[y1: y2 + 1, x1: x2 + 1, :]
        ori_h = h * height
        ratio = new_height / ori_h
        dst_w = int(w * width * ratio)
        if dst_w < 200 or dst_w > 3 * new_height:
            continue
        crop_img = cv2.resize(crop_img, (dst_w, int(new_height)))
        cropped_img.append(crop_img)
        norm_bboxes.append([str(x) for x in [cls_id, *bbox, _]])
        bboxes.append([str(x) for x in [cls_id, x1, y1, x2, y2, _]])
    
    return cropped_img, norm_bboxes, bboxes


def panel_mark_post_process(panel_mark_preds, panel_img_names, norm_bboxes, bboxes):
    """
    1. 过滤非法情况(未检测到表盘中心点, 检测数量少于3个)
    """
    panel_imgs, results_panel_mark = panel_mark_preds["input_imgs"], panel_mark_preds["preds"]
    new_panel_paths, new_results_panel_mark, new_panel_imgs = [], [], []
    new_norm_bboxes, new_bboxes = [], []
    # import pdb
    # pdb.set_trace()
    for i, pred in enumerate(results_panel_mark):
        panel_img_name = panel_img_names[i]
        idx = np.where(pred[:, 0] == CLS_PANEL_CENTER)
        if len(idx[0]) == 0 or len(pred) < 3:
            continue
        new_panel_paths.append(panel_img_name)
        new_results_panel_mark.append(pred)
        new_panel_imgs.append(panel_imgs[i])
        new_norm_bboxes.append(norm_bboxes[i])
        new_bboxes.append(bboxes[i])
    
    return new_panel_paths, new_results_panel_mark, new_panel_imgs, new_norm_bboxes, new_bboxes


def panel_pointer_post_process(results_panel_pointer, kernel_size=3):
    """
    1. 腐蚀和维度缩减
    2. 获取半指针中点
    :param results_panel_pointer:
    :param kernel_size:
    :return:
    """
    new_masks = []
    # each element is [cx, cy]
    semi_pointer_centers = []
    for mask in results_panel_pointer["preds"]:
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.erode(mask, kernel)
        mask = mask.squeeze()
        new_masks.append(mask)
        
        ##########################################
        # 获取半指针中点
        ##########################################
        semi_pointer_cx, semi_pointer_cy = get_one_semi_pointer_center(mask)
        # print("semi_pointer: ", semi_pointer_cx, semi_pointer_cy)
        semi_pointer_centers.append([semi_pointer_cx, semi_pointer_cy])
    
    results_panel_pointer["preds"] = new_masks
    return semi_pointer_centers


def mask_post_process(mask):
    mask_copy = mask.copy()
    tmp_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(mask, tmp_mask, (0, 0), 255)
    img_inverse = cv2.bitwise_not(mask)
    dst = cv2.bitwise_or(mask_copy, img_inverse)
    return dst


def get_one_semi_pointer_center(mask):
    # contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # contour_list = []
    # for i in range(len(contours)):
    #     contour = contours[i]
    #     area = cv2.contourArea(contour)
    #     if area < 200:
    #         continue
    #     contour_list.append(contour)
    # contour = np.vstack(contour_list)
    # # 寻找凸包并绘制凸包（轮廓）
    # hull = cv2.convexHull(contour)
    # # print(hull)
    # # cv2.polylines(mask, [hull], True, (255, 255, 255), 3)
    # # cv2.imshow("mask", mask)
    # # cv2.waitKey()
    # x_min, x_max = hull[:, :, 0].min(), hull[:, :, 0].max()
    # y_min, y_max = hull[:, :, 1].min(), hull[:, :, 1].max()
    y_coords, x_coords = np.nonzero(mask)
    x_min = x_coords.min()
    x_max = x_coords.max()
    y_min = y_coords.min()
    y_max = y_coords.max()
    w = x_max - x_min
    h = y_max - y_min
    
    if h > w:
        part1 = int(np.sum(mask[y_min: int(y_min + h / 4), :]))
        part2 = int(np.sum(mask[int(y_max - h / 4): y_max, :]))
        if part1 > part2:
            semi_pointer_center_y = y_max - h / 4
            sub_mask = mask[int(semi_pointer_center_y): y_max, :]
            part3 = int(np.sum(sub_mask[:, x_min: int(x_min + w / 4)]))
            part4 = int(np.sum(sub_mask[:, int(x_max - w / 4): x_max]))
            if part3 > part4:
                semi_pointer_center_x = x_min + w / 4
            else:
                semi_pointer_center_x = x_max - w / 4
        else:
            semi_pointer_center_y = y_min + h / 4
            sub_mask = mask[y_min: int(semi_pointer_center_y), :]
            part3 = int(np.sum(sub_mask[:, x_min: int(x_min + w / 4)]))
            part4 = int(np.sum(sub_mask[:, int(x_max - w / 4): x_max]))
            if part3 > part4:
                semi_pointer_center_x = x_min + w / 4
            else:
                semi_pointer_center_x = x_max - w / 4
    else:
        part1 = int(np.sum(mask[:, x_min: int(x_min + w / 4)]))
        part2 = int(np.sum(mask[:, int(x_max - w / 4): x_max]))
        if part1 > part2:
            semi_pointer_center_x = x_max - w / 4
            sub_mask = mask[:, int(semi_pointer_center_x): x_max]
            part3 = int(np.sum(sub_mask[y_min: int(y_min + h / 4), :]))
            part4 = int(np.sum(sub_mask[int(y_max - h / 4): y_max, :]))
            if part3 > part4:
                semi_pointer_center_y = y_min + h / 4
            else:
                semi_pointer_center_y = y_max - h / 4
        else:
            semi_pointer_center_x = x_min + w / 4
            sub_mask = mask[:, x_min: int(semi_pointer_center_x)]
            part3 = int(np.sum(sub_mask[y_min: int(y_min + h / 4), :]))
            part4 = int(np.sum(sub_mask[int(y_max - h / 4): y_max, :]))
            if part3 > part4:
                semi_pointer_center_y = y_min + h / 4
            else:
                semi_pointer_center_y = y_max - h / 4
    
    # print(x_min, x_max, y_min, y_max, mask_width, mask_height)
    # part1 = int(np.sum(mask[y_min: int(y_min + h / 4), x_min: int(x_min + w / 4)]))
    # part2 = int(np.sum(mask[int(y_max - h / 4): y_max, int(x_max - w / 4): x_max]))
    # part3 = int(np.sum(mask[int(y_max - h / 4): y_max, x_min: int(x_min + w / 4)]))
    # part4 = int(np.sum(mask[y_min: int(y_min + h / 4), int(x_max - w / 4): x_max]))
    # import pdb
    # pdb.set_trace()
    # if part1 > 5:
    #     if part1 < part2:
    #         semi_pointer_center_x = x_min + w / 4
    #         semi_pointer_center_y = y_min + h / 4
    #     else:
    #         semi_pointer_center_x = x_max - w / 4
    #         semi_pointer_center_y = y_max - h / 4
    # else:
    #     if part3 < part4:
    #         semi_pointer_center_x = x_min + w / 4
    #         semi_pointer_center_y = y_max - h / 4
    #     else:
    #         semi_pointer_center_x = x_max - w / 4
    #         semi_pointer_center_y = y_min + h / 4
    # print(semi_pointer_center_x, semi_pointer_center_y, mask.shape)
    return semi_pointer_center_x, semi_pointer_center_y


def find_one_start_end_panel_mark(panel_mark_results, semi_pointer_center):
    semi_pointer_cx, semi_pointer_cy = semi_pointer_center
    dist = (panel_mark_results[:, 1] - semi_pointer_cx) ** 2 + \
           (panel_mark_results[:, 2] - semi_pointer_cy) ** 2
    # print("Euclidean distance: ", dist)
    sorted_idx = np.argsort(dist)
    mark_start_cls, mark_start_cx, mark_start_cy = panel_mark_results[sorted_idx[0], : 3]
    mark_end_cls, mark_end_cx, mark_end_cy = panel_mark_results[sorted_idx[1], : 3]
    if mark_start_cls > mark_end_cls:
        mark_tmp_cls = mark_start_cls
        mark_start_cls = mark_end_cls
        mark_end_cls = mark_tmp_cls
        
        mark_tmp_cx = mark_start_cx
        mark_start_cx = mark_end_cx
        mark_end_cx = mark_tmp_cx
        
        mark_tmp_cy = mark_start_cy
        mark_start_cy = mark_end_cy
        mark_end_cy = mark_tmp_cy
    mark_start_cls = int(mark_start_cls)
    mark_end_cls = int(mark_end_cls)
    # print("start: ", mark_start_cls, mark_start_cx, mark_start_cy)
    # print("end: ", mark_end_cls, mark_end_cx, mark_end_cy)
    return [mark_start_cls, mark_start_cx, mark_start_cy], [mark_end_cls, mark_end_cx, mark_end_cy]


def find_start_end_panel_mark(new_panel_imgs, results_panel_mark, semi_pointer_centers):
    assert len(new_panel_imgs) == len(results_panel_mark)
    assert len(results_panel_mark) == len(semi_pointer_centers)
    results = {}
    # each element is a list, [cls, cx, cy], not normalized
    results["start_mark_centers"] = []
    # each element is a list, [cls, cx, cy], not normalized
    results["end_mark_centers"] = []
    # each element is a list, [cx, cy], not normalized
    results["panel_centers"] = []
    # each element is a list, [cx, cy], not normalized
    results["semi_pointer_centers"] = []
    for i in range(len(new_panel_imgs)):
        # print("img_name: ", ori_panel_paths[i])
        img = new_panel_imgs[i]
        panel_mark_results = np.array(results_panel_mark[i])
        img_height, img_width, _ = img.shape
        # print("img.shape: ", img_height, img_width)
        panel_mark_results[:, 1::2] *= img_width
        panel_mark_results[:, 2::2] *= img_height
        
        semi_pointer_cx = semi_pointer_centers[i][0]
        semi_pointer_cy = semi_pointer_centers[i][1]
        # import pdb
        # pdb.set_trace()
        # print("panel_mark_results: ", panel_mark_results)
        panel_mark_cls = panel_mark_results[:, 0]
        idx = np.where(panel_mark_cls == CLS_PANEL_CENTER)[0][0]
        panel_cx, panel_cy = panel_mark_results[idx, 1: 3]
        
        panel_inner_mark_results = panel_mark_results[np.where(panel_mark_cls <= MAX_INNER_MARK_CLS)]
        panel_outer_mark_results = panel_mark_results[np.where((panel_mark_cls >= MIN_OUTER_MARK_CLS) |
                                                               (panel_mark_cls == 0))]
        inner_mark_start_info, inner_mark_end_info = find_one_start_end_panel_mark(panel_inner_mark_results,
                                                                                   [semi_pointer_cx, semi_pointer_cy])
        results["start_mark_centers"].append([inner_mark_start_info])
        results["end_mark_centers"].append([inner_mark_end_info])
        
        if len(panel_outer_mark_results) > 1:
            outer_mark_start_info, outer_mark_end_info = find_one_start_end_panel_mark(panel_outer_mark_results,
                                                                                       [semi_pointer_cx,
                                                                                        semi_pointer_cy])
            results["start_mark_centers"].append(outer_mark_start_info)
            results["end_mark_centers"].append(outer_mark_end_info)
        results["panel_centers"].append([panel_cx, panel_cy])
        results["semi_pointer_centers"].append([semi_pointer_cx, semi_pointer_cy])
    
    return results


def point2vector(start_point, end_point):
    vec = np.array(end_point) - np.array(start_point)
    return vec


def calc_angle(v1, v2):
    vector_dot_product = np.dot(v1, v2)
    arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    angle = np.degrees(arccos)
    return angle


def calc_reading_num(start_end_panel_marks):
    # each element is a list, [cls, cx, cy], not normalized
    start_mark_centers = start_end_panel_marks["start_mark_centers"]
    # each element is a list, [cls, cx, cy], not normalized
    end_mark_centers = start_end_panel_marks["end_mark_centers"]
    # each element is a list, [cx, cy], not normalized
    panel_centers = start_end_panel_marks["panel_centers"]
    semi_pointer_centers = start_end_panel_marks["semi_pointer_centers"]
    
    reading_nums = []
    # traverse number of panel_centers
    for i in range(len(panel_centers)):
        # import pdb
        # pdb.set_trace()
        each_reading_nums = []
        # traverse panel inner and outer mark
        for j in range(len(start_mark_centers[i])):
            mark_start_cls, mark_start_cx, mark_start_cy = start_mark_centers[i][j]
            mark_end_cls, mark_end_cx, mark_end_cy = end_mark_centers[i][j]
            panel_center = panel_centers[i]
            semi_pointer_point = semi_pointer_centers[i]
            
            mark_start_vec = point2vector(panel_center, [mark_start_cx, mark_start_cy])
            mark_end_vec = point2vector(panel_center, [mark_end_cx, mark_end_cy])
            semi_pointer_vec = point2vector(panel_center, semi_pointer_point)
            angle1 = calc_angle(mark_start_vec, mark_end_vec)
            angle3 = calc_angle(semi_pointer_vec, mark_end_vec)
            angle4 = calc_angle(semi_pointer_vec, mark_start_vec)
            # import pdb
            # pdb.set_trace()
            if angle3 <= angle1 and angle4 <= angle1:
                angle2 = calc_angle(mark_start_vec, semi_pointer_vec)
                percent = angle2 / angle1
            else:
                percent = 0
            if abs(percent) < 0.1:
                percent = 0
            start_reading = MARK_LABEL_MAP[str(mark_start_cls)]
            mark_gap = abs(MARK_LABEL_MAP[str(mark_end_cls)] - MARK_LABEL_MAP[str(mark_start_cls)])
            reading_num = start_reading + mark_gap * percent
            each_reading_nums.append(reading_num)
        
        reading_nums.append(each_reading_nums)
    
    return reading_nums


def recog_one_img_reading(source_ori,
                          model_det_piezometer, model_det_panel_mark, model_seg_panel_pointer,
                          kernel_size=3, save_path="tmp_results/piezometer_reading",
                          show_temp=False, device="cuda:0"):
    """
    :param source_ori: 输入图片的绝对路径
    :param model_det_piezometer: 表计检测模型
    :param model_det_panel_mark: 表计表盘刻度检测模型
    :param model_seg_panel_pointer: 表计表盘指针分割模型
    :param kernel_size: 对分割结果进行腐蚀操作的核大小，为1时，不进行腐蚀操作
    :param save_path: 结果保存路径，默认保存在工程目录下的tmp_results/piezometer_reading
    :param show_temp: 是否保存算法临时结果，用于debug，True表示保存，False表示不保存
    :param device: 推理设备，默认gpu
    :return:
    reading_result: json格式
    {
        "input_info": {
            "height": 整数，图像高度
            "width": 整数，图像宽度
            "img_path": 字符串，图像路径
        }
        "output_info": {
            "norm_bbox": [[cls_id, norm_cx, norm_cy, norm_w, norm_h, conf],
                          [cls_id, norm_cx, norm_cy, norm_w, norm_h, conf], ...]
                         表计在图像中的相对位置，二维列表，每个元素是一个长度为6的列表，依次表示:
                         类别id, 归一化后矩形框中心点x坐标, 归一化后矩形框中心点y坐标, 归一化后矩形框宽, 归一化后矩形框高, 置信度
            "bbox": [[cls_id, cx, cy, w, h, conf],
                     [cls_id, cx, cy, w, h, conf], ...]
                    表计在图像中的绝对位置，二维列表，每个元素是一个长度为6的列表，依次表示:
                    类别id, 矩形框中心点x坐标, 矩形框中心点y坐标, 矩形框宽, 矩形框高, 置信度
            "reading": [[表计读数1, 表计读数2], [表计读数1], ...]，
                       表计读数结果，二维列表，每个元素是一个列表，表示表计读数结果，读数结果为浮点数，可能为2个或者1个
        }
    }
    """
    reading_result = {}
    reading_result["input_info"] = {}
    reading_result["output_info"] = {}
    
    img0, filename = read_img(source_ori)
    ori_h, ori_w, _ = img0.shape
    imgsz = 320
    img = resize_img(img0, ratio=(imgsz / ori_h))
    
    reading_result["input_info"]["img_path"] = source_ori
    reading_result["input_info"]["height"] = ori_h
    reading_result["input_info"]["width"] = ori_w
    
    # import pdb
    # pdb.set_trace()
    if show_temp:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
            print("%s exist! Delete!" % save_path)
    ##########################################
    # get piezometer results
    ##########################################
    det_piezometer_t0 = time.time()
    piezometer_preds = run_detector(model_det_piezometer,  # model.pt path(s)
                                    source=[img],  # file/dir/URL/glob, 0 for webcam
                                    imgsz=imgsz,  # inference size (pixels)
                                    conf_thres=0.25,  # confidence threshold
                                    iou_thres=0.45,  # NMS IOU threshold
                                    max_det=1000,  # maximum detections per image
                                    device=device,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                                    )
    det_piezometer_t1 = time.time()
    det_piezometer_time = det_piezometer_t1 - det_piezometer_t0
    
    if show_temp:
        piezometer_save_path = "%s/piezometer" % save_path
        os.makedirs(piezometer_save_path, exist_ok=True)
        save_names = ["%s/%s.png" % (piezometer_save_path, filename)]
        vis_det_results(piezometer_preds, save_names, line_thickness=3)
    
    # import pdb
    # pdb.set_trace()
    # 未检测到表计时，直接返回
    if len(piezometer_preds["preds"]) == 0:
        reading_result["output_info"]["norm_bbox"] = []
        reading_result["output_info"]["bbox"] = []
        reading_result["output_info"]["reading"] = []
        return reading_result
    
    ##########################################
    # 根据表计检测结果裁剪，得到表盘
    ##########################################
    det_piezometer_post_t0 = time.time()
    panel_imgs, panel_norm_bboxes, panel_bboxes = piezometer_post_process(piezometer_preds["preds"][0], img0,
                                                                          new_height=imgsz)
    det_piezometer_post_t1 = time.time()
    det_piezometer_post_time = det_piezometer_post_t1 - det_piezometer_post_t0
    
    panel_save_root = "%s/piezometer_panel" % save_path
    panel_img_names = ["%s/%s_%s.png" % (panel_save_root, filename,
                                         str(idx).zfill(2)) for idx in range(len(panel_imgs))]
    if show_temp:
        os.makedirs(panel_save_root, exist_ok=True)
        write_img(panel_img_names, panel_imgs)
    
    # import pdb
    # pdb.set_trace()
    ##########################################
    # 获取表盘上mark的检测结果
    ##########################################
    """
    {img_paths: [img_path], preds: [[[cls, cx, cy, w, h, conf], [cls, cx, cy, w, h, conf]]]}
    """
    det_panel_mark_t0 = time.time()
    panel_mark_preds = run_detector(model_det_panel_mark,  # model.pt path(s)
                                    source=panel_imgs,
                                    imgsz=imgsz,  # inference size (pixels)
                                    conf_thres=0.6,  # confidence threshold
                                    iou_thres=0.45,  # NMS IOU threshold
                                    max_det=1000,  # maximum detections per image
                                    device=device,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                                    )
    det_panel_mark_t1 = time.time()
    det_panel_mark_time = det_panel_mark_t1 - det_panel_mark_t0
    
    if show_temp:
        panel_mark_save_root = "%s/piezometer_panel_mark" % save_path
        os.makedirs(panel_mark_save_root, exist_ok=True)
        panel_mark_img_names = ["%s/%s" % (panel_mark_save_root,
                                           os.path.basename(name)) for name in panel_img_names]
        vis_det_results(panel_mark_preds, panel_mark_img_names, line_thickness=3)
    
    det_panel_mark_post_t0 = time.time()
    new_panel_paths, new_results_panel_mark, new_panel_imgs, new_norm_bboxes, new_bboxes \
        = panel_mark_post_process(panel_mark_preds, panel_img_names, panel_norm_bboxes, panel_bboxes)
    reading_result["output_info"]["norm_bbox"] = new_norm_bboxes
    reading_result["output_info"]["bbox"] = new_bboxes
    det_panel_mark_post_t1 = time.time()
    det_panel_mark_post_time = det_panel_mark_post_t1 - det_panel_mark_post_t0
    
    ##########################################
    # 获取表盘指针分割结果
    ##########################################
    """
    results_panel_pointer: dict
    img_paths: a list, piezometer panel paths
    preds: a list, each element is a predicted mask
    """
    seg_pointer_t0 = time.time()
    results_panel_pointer = run_segmentator(model_seg_panel_pointer, new_panel_imgs)
    seg_pointer_t1 = time.time()
    seg_pointer_time = seg_pointer_t1 - seg_pointer_t0
    
    if show_temp:
        panel_pointer_save_root = "%s/piezometer_panel_pointer" % save_path
        os.makedirs(panel_pointer_save_root, exist_ok=True)
        mask_save_paths = ["%s/%s" % (panel_pointer_save_root, os.path.basename(name))
                           for name in new_panel_paths]
        vis_seg_results(results_panel_pointer, mask_save_paths)
    
    pointer_post_t0 = time.time()
    semi_pointer_centers = panel_pointer_post_process(results_panel_pointer, kernel_size)
    pointer_post_t1 = time.time()
    seg_pointer_post_time = pointer_post_t1 - pointer_post_t0
    
    if show_temp:
        panel_pointer_erode_save_root = "%s/piezometer_panel_pointer_erode%d" % (save_path, kernel_size)
        os.makedirs(panel_pointer_erode_save_root, exist_ok=True)
        mask_save_paths = ["%s/%s" % (panel_pointer_erode_save_root, os.path.basename(name))
                           for name in new_panel_paths]
        vis_seg_results(results_panel_pointer, mask_save_paths)
    
    ##########################################
    # 找到距离半指针中点最近的刻度标
    ##########################################
    calc_t0 = time.time()
    start_end_panel_marks = find_start_end_panel_mark(new_panel_imgs, new_results_panel_mark,
                                                      semi_pointer_centers)
    
    ##########################################
    # 计算读数
    ##########################################
    reading_nums = calc_reading_num(start_end_panel_marks)
    reading_result["output_info"]["reading"] = reading_nums
    calc_t1 = time.time()
    calc_time = calc_t1 - calc_t0
    
    total_time = det_piezometer_time + det_piezometer_post_time + \
                 det_panel_mark_time + det_panel_mark_post_time + \
                 seg_pointer_time + seg_pointer_post_time + \
                 calc_time
    
    print("total_time: %.3f; run_det_piezometer_time: %.3f; run_det_piezometer_post_time: %.3f; "
          "run_det_panel_mark_time: %.3f; run_det_panel_mark_post_time: %.3f; "
          "run_seg_pointer_time: %.3f; run_seg_post_time: %.3f; run_calc_time: %.3f" % (
              total_time, det_piezometer_time, det_piezometer_post_time,
              det_panel_mark_time, det_panel_mark_post_time,
              seg_pointer_time, seg_pointer_post_time, calc_time
          ))
    
    if show_temp:
        reading_res_save_path = "%s/result" % save_path
        os.makedirs(reading_res_save_path, exist_ok=True)
        for j in range(len(new_panel_imgs)):
            img = new_panel_imgs[j]
            panel_mark_results = np.array(new_results_panel_mark[j])
            img_height, img_width, _ = img.shape
            # print("img.shape: ", img_height, img_width)
            panel_mark_results[:, 1::2] *= img_width
            panel_mark_results[:, 2::2] *= img_height
            semi_pointer_cx, semi_pointer_cy = start_end_panel_marks["semi_pointer_centers"][j]
            # import pdb
            # pdb.set_trace()
            for res in panel_mark_results:
                cls = str(MARK_LABEL_MAP[str(int(res[0]))])
                x = int(res[1])
                y = int(res[2])
                cv2.putText(img, cls, (x, y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
                cv2.circle(img, (x, y), 1, (0, 0, 255), 3)
            cv2.circle(img, (int(semi_pointer_cx), int(semi_pointer_cy)), 3, (255, 0, 0), 3)
            for i in range(len(reading_nums[j])):
                reading_num = reading_nums[j][i]
                cv2.putText(img, str(round(reading_num, 2)), (10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 1)
            base_name = os.path.basename(new_panel_paths[j])
            cv2.imwrite("%s/%s" % (reading_res_save_path, base_name), img)
    
    return reading_result


def save_reading_result(reading_result, save_path):
    img_path = reading_result["input_info"]["img_path"]
    img = cv2.imread(img_path)
    bboxes = reading_result["output_info"]["bbox"]
    readings = reading_result["output_info"]["reading"]
    # 每个图片的表计位置
    for i in range(len(bboxes)):
        # 每个表计的位置信息
        cls_id, x1, y1, x2, y2, conf = bboxes[i]
        x1, y1, x2, y2 = [int(x) for x in [x1, y1, x2, y2]]
        # print(img_path, x1, y1, x2, y2)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        reading = readings[i]
        # 每个表计的读数结果
        for j in range(len(reading)):
            img = cv2.putText(img, str(round(reading[j], 2)), (x1 - 10, y1 - (j + 1) * 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
    os.makedirs(save_path, exist_ok=True)
    base_name = os.path.basename(img_path)
    img_save_path = "%s/%s" % (save_path, base_name)
    cv2.imwrite(img_save_path, img)
    
    # json_str = json.dumps(reading_result, indent=4, ensure_ascii=False)
    # with open("%s/result.json" % save_path, "w", encoding="utf-8") as f:
    #     f.write(json_str)


def recog_img_reading1(model_det_piezometer, model_det_panel_mark, model_seg_panel_pointer,
                       abso_img_paths, save_path, kernel_size=1, device="cuda", show_temp=False):
    """
    :param abso_img_paths: 输入图片的绝对路径，字符串列表
    :param save_path: 结果保存路径
    :param kernel_size: 对分割结果进行腐蚀操作的核大小，为1时，不进行腐蚀操作
    :param device: 推理设备，默认gpu
    :param show_temp: 是否保存算法临时结果，用于debug，True表示保存，False表示不保存
    :return:
    results: 列表，每个元素表示对应图片的识别结果，结果格式见recog_one_img_reading方法的返回结果
    """
    
    results = []
    for source_ori in tqdm(abso_img_paths):
        save_path_ = "%s/%s" % (save_path, os.path.basename(source_ori).split(".")[0])
        if not os.path.exists(source_ori):
            print("%s not exist" % source_ori)
            continue
        reading_result = recog_one_img_reading(source_ori,
                                               model_det_piezometer,
                                               model_det_panel_mark,
                                               model_seg_panel_pointer, kernel_size,
                                               "%s/alg_out" % save_path_, show_temp, device)
        results.append(reading_result)
        if show_temp:
            # 保存识别结果
            save_reading_result(reading_result, "%s/result" % save_path_)
    return results


@app.route("/api/piezometer_reading", methods=["POST"])
def piezometer_reading():
    data = request.json
    keys = list(data.keys())
    error_warn_info = ""
    if "img_root" in keys:
        img_root = data["img_root"]
    else:
        error_warn_info += "错误: 缺少参数img_root. "
    if "save_path" in keys:
        save_path = data["save_path"]
    else:
        error_warn_info += "警告: 缺少参数save_path，使用默认值./tmp_results/piezometer. "
        save_path = "./tmp_results/piezometer"
    if "kernel_size" in keys:
        kernel_size = float(data["kernel_size"])
    else:
        error_warn_info += "警告: kernel_size，使用默认值1. "
        kernel_size = 1
    if "device" in keys:
        device = data["device"]
    else:
        error_warn_info += "警告: 缺少参数device，使用默认值cuda. "
        device = "cuda"
    if "show_temp" in keys:
        show_temp = bool(data["show_temp"])
    else:
        error_warn_info += "警告: 缺少参数show_temp，使用默认值，False. "
        show_temp = False
    
    res_dict = {}
    res_dict["error_warn_info"] = error_warn_info
    if "错误" in error_warn_info:
        res_dict["results"] = []
    else:
        abso_img_paths = sorted(glob("%s/*" % img_root))
        results = recog_img_reading1(model_det_piezometer, model_det_panel_mark, model_seg_panel_pointer,
                                     abso_img_paths, save_path, kernel_size, device, show_temp)
        res_dict["results"] = results
    
    response = make_response(jsonify(res_dict))
    response.headers["Content-Type"] = "application/json;charset=UTF-8"
    return response


if __name__ == '__main__':
    """
    1. 启动服务器
    cd 至代码根目录
    python main_piezometer_reading_web.py
    2. 客户端接口调用
    表计读数api调用: curl -X POST http://10.10.3.99:9997/api/piezometer_reading -d "{\"img_root\": \"/home/dykj/work/IndustryQualityTesting/test_imgs/piezometer1\", \"save_path\": \"/home/dykj/work/IndustryQualityTesting/tmp_save/piezometer1\", \"kernel_size\": 1, \"device\": \"cuda\", \"show_temp\": 1}" -H "Content-Type:application/json"
    0: False, 1: True
    """
    ckpt_det_piezometer = "./ckpts/piezometer_det.pt"
    ckpt_det_panel_mark = "./ckpts/piezometer_panel_mark_det.pt"
    config_seg_pointer = "./ckpts/piezometer_pointer_seg.py"
    ckpt_seg_pointer = "./ckpts/piezometer_pointer_seg.pth"
    device = "cuda"
    model_det_piezometer = init_model_detector(ckpt_det_piezometer, device)
    model_det_panel_mark = init_model_detector(ckpt_det_panel_mark, device)
    model_seg_panel_pointer = init_model_segmentator(config_seg_pointer, ckpt_seg_pointer, device)
    app.run(host="0.0.0.0", port=9997, debug=True)
