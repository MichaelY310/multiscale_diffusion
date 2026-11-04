#!/bin/bash
# Train RDD-FM multistage from scratch on original ImageNet (ADM center_crop_arr).
# Flow Matching version with correct time-axis partitioning.
# 32x32→DiT-XL/2, 16x16→DiT-L/2, 8x8→DiT-B/2.
# Stage mapping: coarsest(8x8) t∈[0,0.3), middle(16x16) t∈[0.3,0.6), finest(32x32) t∈[0.6,1.0)
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
RESULTS_DIR=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original/RDD-FM-Multistage-300-300-400

torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=29610 \
    /home/tiger/multiscale_diffusion/2d/train_RDD_FM_imagenet256.py \
    --model-32           DiT-XL/2 \
    --model-16           DiT-L/2 \
    --model-8            DiT-B/2 \
    --stage-lengths      300,300,400 \
    --diffusion-steps    1000 \
    --results-dir        ${RESULTS_DIR} \
    --global-batch-size  256 \
    --num-workers        4 \
    --epochs             1400 \
    --log-every          100 \
    --ckpt-every         5000 \
    --vae                ema \
    --global-seed        0 \
    --data-source        imagenet \
    --data-path          ${DATA_ROOT}/ILSVRC/Data/CLS-LOC/train \
    --resume-32          /mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original/RDD-FM-Multistage-300-300-400/imagenet256-fm-multiscale/checkpoints_32x32/0060000.pt \
    --resume-16          /mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original/RDD-FM-Multistage-300-300-400/imagenet256-fm-multiscale/checkpoints_16x16/0060000.pt \
    --resume-8           /mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original/RDD-FM-Multistage-300-300-400/imagenet256-fm-multiscale/checkpoints_8x8/0060000.pt
