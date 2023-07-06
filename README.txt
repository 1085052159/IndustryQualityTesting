1. 环境配置
见SegFormer下的README.md和yolov5下的RADDME.md，两个运行环境都需要

2. 运行
main_bolt_loosen.py: 螺丝松动检测
main_piezometer_reading.py: 表计读数预测
main_oil_level.py: 油箱液位预测
web_interface.py: web url方式调用接口
view_results.py: 将识别结果复制到同一个文件以便快速查看运行结果的脚本

3. 目录说明
SegFormer & yolov5: 代码库文件
test_imgs: 用于测试表计读数和螺丝松动的图片
*.py: 见2

4. 部署说明
1. 将work_dirs拷贝至SegFormer根目录
2. 将pretrained拷贝至SegFormer根目录(训练时需要)
3. 将yolov5s.pt和yolov5m.pt拷贝至yolov5根目录(训练时需要)
4. 将runs拷贝至yolov5根目录
5. 将test_imgs拷贝至项目根目录
