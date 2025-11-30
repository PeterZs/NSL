#!/bin/bash
mode=$1
if [ $# -ge 2 ]; then
    init_flow=$2
else
    init_flow=no
fi
if [ $# -ge 3 ]; then
    key=$3
    pat=$4
else
    key=None
    pat=None
fi

if [[ "${mode}" == lp ]]; then
    ckpt=exp/left_patt/ckpts/101_epoch_raft-stereo.pth.gz
    ds_cfgs=cfgs/train_local_nolr.yaml
elif [[ "${mode}" == lr_gray ]]; then
    ckpt=exp/left_right/ckpts/101_epoch_raft-stereo.pth.gz
    ds_cfgs=cfgs/train_local_matchlr.yaml
elif [[ "${mode}" == lr_rgb ]]; then
    ckpt=exp/left_right_rgb/ckpts/101_epoch_raft-stereo.pth.gz
    ds_cfgs=cfgs/train_local_matchlr_rgb.yaml
fi


python test.py --valid_iters 32 \
    --restore_ckpt $ckpt \
    --dataset_cfgs $ds_cfgs \
    --n_downsample 2 \
    --mixed_precision \
    --output_directory ./debug \
    --init_flow $init_flow \
    --key $key \
    --patt $pat