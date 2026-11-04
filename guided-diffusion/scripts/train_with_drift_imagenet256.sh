#!/bin/bash
# Train multi-scale UNet on ImageNet-1k-256×256 with drift.
#
# Architecture per resolution:
#   default: three separate UNets
#   optional: one shared 256×256 UNet + adapters for 128×128 / 64×64
#
# Dataset: benjamin-paine/imagenet-1k-256x256 (HuggingFace)
# On first run the dataset will be downloaded (~28 GB) to HF_DATASETS_CACHE.
#
# Checkpoints are saved under:
#   guided_diffusion/results_imagenet256/<exp_id>-imagenet256-multiscale-drift/
#       checkpoints_256x256/<step>.pt
#       checkpoints_128x128/<step>.pt
#       checkpoints_64x64/<step>.pt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

source "$REPO_DIR/.venv/bin/activate"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_DEBUG=WARN
export HF_DATASETS_CACHE=/scr/yuheng/DevOpsBench/multiscale_diffusion/.hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RESULTS_DIR="/scr/yuheng/DevOpsBench/multiscale_diffusion/guided-diffusion/results_with_drift_imagenet256"

# ---- Latest checkpoints for resume (leave empty to train from scratch) ----
PREV_CKPT=
CKPT_STEP=
USE_SINGLE_MODEL=0

mkdir -p "$RESULTS_DIR"

MPIEXEC=/usr/mpi/gcc/openmpi-4.1.5rc2/bin/mpiexec

# Build optional resume flags
RESUME_FLAGS=""
if [ -n "$PREV_CKPT" ] && [ -n "$CKPT_STEP" ]; then
    if [ "$USE_SINGLE_MODEL" = "1" ]; then
        RESUME_FLAGS="--resume_shared ${PREV_CKPT}/checkpoints_shared/${CKPT_STEP}.pt"
    else
        RESUME_FLAGS="--resume_256 ${PREV_CKPT}/checkpoints_256x256/${CKPT_STEP}.pt \
                      --resume_128 ${PREV_CKPT}/checkpoints_128x128/${CKPT_STEP}.pt \
                      --resume_64  ${PREV_CKPT}/checkpoints_64x64/${CKPT_STEP}.pt"
    fi
fi

MODEL_MODE_FLAG=""
if [ "$USE_SINGLE_MODEL" = "1" ]; then
    MODEL_MODE_FLAG="--use_single_model"
fi

$MPIEXEC -n 4 -x CUDA_VISIBLE_DEVICES \
    python "$SCRIPT_DIR/image_train_with_drift_imagenet256.py" \
        --results_dir       "$RESULTS_DIR"  \
        --image_size        256             \
        --total_steps       1000000         \
        --batch_size        48              \
        --num_workers       4               \
        --lr                1e-4            \
        --weight_decay      0.0             \
        --ema_rate          0.9999          \
        --log_every         100             \
        --ckpt_every        5000            \
        --noise_schedule    linear          \
        --num_res_blocks    2               \
        --num_heads         4               \
        --num_channels_256  192             \
        --num_channels_128  128             \
        --num_channels_64   64              \
        --attention_resolutions_256 "32,16,8" \
        --attention_resolutions_128 "32,16,8" \
        --attention_resolutions_64  "16,8"    \
        --use_checkpoint                    \
        --sample_scale      2               \
        $MODEL_MODE_FLAG                    \
        $RESUME_FLAGS

# Run in background:
# nohup bash "$SCRIPT_DIR/train_with_drift_imagenet256.sh" \
#     > "$REPO_DIR/train_with_drift_imagenet256.log" 2>&1 &
