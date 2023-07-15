import sys
import os

from main_oil_level import recog_img_oil_level

sys.path.append(os.getcwd() + "/yolov5")
sys.path.append(os.getcwd() + "/SegFormer")

from glob import glob
from flask import Flask, request, jsonify, make_response

from main_bolt_loosen import recog_img_bolt_state
from main_piezometer_reading import recog_img_reading

app = Flask(__name__)
app.config.from_object(__name__)
app.config["JSON_AS_ASCII"] = False


@app.route("/api/bolt_loosen", methods=["POST"])
def bolt_loosen():
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
        error_warn_info += "警告: 缺少参数save_path，使用默认值./tmp_results/bolt. "
        save_path = "./tmp_results/bolt"
    if "thresh_angle" in keys:
        thresh_angle = float(data["thresh_angle"])
    else:
        error_warn_info += "警告: 缺少参数thresh_angle，使用默认值5. "
        thresh_angle = 5
    if "thresh_dist" in keys:
        thresh_dist = float(data["thresh_dist"])
    else:
        error_warn_info += "警告: 缺少参数thresh_dist，使用默认值5. "
        thresh_dist = 5
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
        results = recog_img_bolt_state(abso_img_paths, save_path,
                                       thresh_angle, thresh_dist,
                                       device, show_temp)
        res_dict["results"] = results
    response = make_response(jsonify(res_dict))
    response.headers["Content-Type"] = "application/json;charset=UTF-8"
    return response


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
        results = recog_img_reading(abso_img_paths, save_path, kernel_size, device, show_temp)
        res_dict["results"] = results
    
    response = make_response(jsonify(res_dict))
    response.headers["Content-Type"] = "application/json;charset=UTF-8"
    return response


@app.route("/api/oil_level_reading", methods=["POST"])
def oil_level_reading():
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
        error_warn_info += "警告: 缺少参数save_path，使用默认值./tmp_results/oil_level. "
        save_path = "./tmp_results/oil_level"
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
        results = recog_img_oil_level(abso_img_paths, save_path, device, show_temp)
        res_dict["results"] = results

    response = make_response(jsonify(res_dict))
    response.headers["Content-Type"] = "application/json;charset=UTF-8"
    return response



if __name__ == '__main__':
    """
    1. 启动服务器
    cd 至代码根目录
    export FLASK_APP=web_interface
    export FLASK_ENV=development
    flask run --host 0.0.0.0
    2. 客户端接口调用
    螺丝松动api调用: curl -X POST http://10.10.3.99:9999/api/bolt_loosen -d "{\"img_root\": \"/home/dykj/work/IndustryQualityTesting/test_imgs/bolt1\", \"save_path\": \"/home/dykj/work/IndustryQualityTesting/tmp_save/bolt1\", \"thresh_angle\": 5, \"thresh_dist\": 5, \"device\": \"cuda\", \"show_temp\": 1}" -H "Content-Type:application/json"
    表计读数api调用: curl -X POST http://10.10.3.99:9999/api/piezometer_reading -d "{\"img_root\": \"/home/dykj/work/IndustryQualityTesting/test_imgs/piezometer1\", \"save_path\": \"/home/dykj/work/IndustryQualityTesting/tmp_save/piezometer1\", \"kernel_size\": 1, \"device\": \"cuda\", \"show_temp\": 1}" -H "Content-Type:application/json"
    油液读数api调用: curl -X POST http://10.10.3.99:9999/api/oil_level_reading -d "{\"img_root\": \"/home/dykj/work/IndustryQualityTesting/test_imgs/oil_level1\", \"save_path\": \"/home/dykj/work/IndustryQualityTesting/tmp_save/oil_level1\", \"device\": \"cuda\", \"show_temp\": 1}" -H "Content-Type:application/json"
    0: False, 1: True
    """
    app.run(debug=True, host="0.0.0.0", post=5000)
