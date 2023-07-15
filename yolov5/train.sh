weights="yolov5s.pt"
cfg="models/yolov5s.yaml"
data="data/hole.yaml"
data="data/oil_level.yaml"
hyp="data/hyps/hyp.scratch.yaml"
epochs="300"
batch_size="16"
img_size="416 416"
workers="0"
multi_scale=""
#multi_scale="--multi_scale"
project="runs/train"
#name="hole_${img_size/ /_}_${epochs}_${batch_size}"
name="oil_level_${img_size/ /_}_${epochs}_${batch_size}"
save_period="-1"

weights_arg="--weights $weights"
cfg_arg="--cfg $cfg"
data_arg="--data $data"
hyp_arg="--hyp $hyp"
epochs_arg="--epochs $epochs"
batch_arg="--batch-size $batch_size"
img_size_arg="--img-size $img_size"
workers_arg="--workers $workers"
multi_scale_arg=$multi_scale
project_arg="--project $project"
name_arg="--name $name"
save_period_arg="--save_period $save_period"

args="$weights_arg $cfg_arg $data_arg $hyp_arg $epochs_arg $batch_arg $img_size_arg $workers_arg $multi_scale_arg $project_arg $name_arg $save_period_arg"
echo $args
python train.py $args
#python train.py --weights yolov5s.pt --cfg models/yolov5s.yaml --data data/hole.yaml --hyp data/hyps/hyp.scratch.yaml --epochs 300 --batch-size 16 --img-size 416 416 --workers 0 --project runs/train --name hole_416_416_300_16 --save_period -1

