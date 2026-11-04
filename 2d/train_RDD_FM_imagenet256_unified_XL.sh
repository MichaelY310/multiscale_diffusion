#!/bin/bash
# Train RDD-FM Unified (DiT-XL/2) from scratch on original ImageNet (ADM center_crop_arr).
# Flow Matching version: Single DiTMultiRes model for all resolutions: 32x32, 16x16, 8x8.
# 8 GPUs, global batch size 256.

set -e

export HF_HUB_OFFLINE=1
unset no_proxy NO_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY HF_ENDPOINT
export HF_HOME=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/.cache/huggingface
export HF_DATASETS_CACHE=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/tmp/hf_datasets_cache
export TMPDIR=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/tmp
source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATA_ROOT=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/data/imagenet
RESULTS_DIR=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original/RDD-FM-Unified-XL

torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=29611 \
    /home/tiger/multiscale_diffusion/2d/train_RDD_FM_imagenet256_unified.py \
    --model              DiT-XL/2 \
    --resolutions        32,16,8 \
    --stage-lengths      300,300,400 \
    --diffusion-steps    1000 \
    --results-dir        ${RESULTS_DIR} \
    --global-batch-size  256 \
    --num-workers        4 \
    --no-pin-memory \
    --epochs             1400 \
    --log-every          100 \
    --ckpt-every         5000 \
    --vae                ema \
    --global-seed        0 \
    --data-source        imagenet \
    --data-path          ${DATA_ROOT}/ILSVRC/Data/CLS-LOC/train \
    --resume             /mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original/RDD-FM-Unified-XL/000-imagenet256-fm-unified/checkpoints/0215000.pt
