weights="runs/train/bolt_416_416_150_16/weights/best.pt"
weights="runs/train/bolt_prelabel_new_anchor_640_640_300_16/weights/best.pt"
weights="runs/train/bolt_416_416_300_32/weights/best.pt"
weights="runs/train/bolt_prelabel_yolov5m_new_anchor_640_640_300_82/weights/best.pt"
weights="runs/train/bolt/weights/best.pt"
source="E:/Downloads/test_imgs"
source="../vid_frames/DJI_20230711104903_0005_V"
source="F:/dataset/bolt_piezometer/bolt2/test_imgs"
vid_name="DJI_20230728101911_0002_V"
source="../vid_frames1/${vid_name}"
imgsz=416
imgsz=640
conf_thres=0.25
#iou_thres=0.45
iou_thres=0.2
max_det=1000
save_txt="--save-txt"
save_crop="--save-crop"
project="runs/detect"
name="bolt_${imgsz}_${conf_thres}_${iou_thres}/${vid_name}"

weights_arg="--weights $weights"
source_arg="--source $source"
imgsz_arg="--imgsz $imgsz"
conf_thres_arg="--conf-thres $conf_thres"
iou_thres_arg="--iou-thres $iou_thres"
max_det_arg="--max-det $max_det"
save_txt_arg=$save_txt
save_crop_arg=$save_crop
project_arg="--project $project"
name_arg="--name $name"

args="$weights_arg $source_arg $imgsz_arg $conf_thres_arg $iou_thres_arg $max_det_arg $save_txt_arg $save_crop_arg $project_arg $name_arg"
echo python detect.py $args
python detect.py $args
# python train.py --weights yolov5s.pt --cfg models/yolov5s.yaml --data data/bolt.yaml --hyp data/hyps/hyp.scratch.yaml --epochs 300 --batch-size 16 --img-size 416 416 --workers 0 --project runs/train --name bolt_416_416_300_16 --save_period -1
