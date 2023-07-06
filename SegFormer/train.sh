config="local_configs/segformer/B0/segformer.b0.256x256.pointer.160k.py"
config="local_configs/segformer/B0/segformer.b0.256x256.bolt_line.2k.py"
config="local_configs/segformer/B0/segformer.b0.256x256.bolt_line.6k.py"
config="local_configs/segformer/B1/segformer.b1.256x256.bolt_line.6k.py"
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=2333 tools/train.py $config --launcher pytorch