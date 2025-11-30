#!/bin/bash
cfgs=$1
gpus=$2
gpus_ids=$3
if [ $# -ge 4 ]; then
    # the 5th arg should be the ckpt to resume.
    resume=$4
else
    # or no resume.
    resume=null
fi

 CUDA_VISIBLE_DEVICES=$gpus_ids python3 -m torch.distributed.launch \
    --nproc_per_node=$gpus \
    --nnodes 1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=20799 \
    train.py --cfgs-path $cfgs --exproot ./ \
    --resume $resume
