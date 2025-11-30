#!/bin/bash

python train_stereo.py --batch_size 8 --train_iters 22 --valid_iters 32 \
    --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 \
    --num_steps 200000 --mixed_precision \
    --deepsl_args_path cfgs/train_local_matchlr.yaml \
    --train_datasets dataset_deepsl \
    --expdir exp/left_right_stable/ \
    --num_workers 3 \
    --steps_per_epoch 2000 \
    --lr 0.0001 \
    --max_grad_norm 0.8