#!/bin/bash
# Train DiT-XL/2 from scratch on original ImageNet (ADM center_crop_arr preprocessing).
# Matches the official DiT paper pipeline exactly.
# 8 GPUs, global batch size 256, 1000 diffusion steps, linear schedule.

set -e

export HF_HUB_OFFLINE=1
unset no_proxy NO_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY HF_ENDPOINT
export HF_HOME=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/.cache/huggingface
export HF_DATASETS_CACHE=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/tmp/hf_datasets_cache
export TMPDIR=/tmp
source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATA_ROOT=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/data/imagenet
RESULTS_DIR=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original/DiT-XL-2

torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=29600 \
    /home/tiger/multiscale_diffusion/DiT/train_imagenet256.py \
    --results-dir        ${RESULTS_DIR} \
    --model              DiT-XL/2 \
    --image-size         256 \
    --num-classes        1000 \
    --global-batch-size  256 \
    --num-workers        4 \
    --no-pin-memory \
    --epochs             1400 \
    --log-every          100 \
    --ckpt-every         10000 \
    --vae                ema \
    --global-seed        0 \
    --data-source        imagenet \
    --data-path          ${DATA_ROOT}/ILSVRC/Data/CLS-LOC/train \
    --resume             /mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original/DiT-XL-2/006-DiT-XL-2/checkpoints/0400000.pt
