#!/bin/bash
# Multi-scale drift sampling: 64×64 → 128×128 → 256×256
#
# Set CKPT_DIR and CKPT_STEP to point at a saved experiment, then run.
# Output is a PNG grid saved next to this script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

cd "$REPO_DIR"
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=4          # single GPU is enough for sampling
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Checkpoint to sample from ----
CKPT_DIR="/scr/yuheng/DevOpsBench/multiscale_diffusion/guided-diffusion/results_with_drift_imagenet256/006-imagenet256-multiscale-drift"
CKPT_STEP=0075000   # 7-digit step, e.g. 0035000

# ---- Output ----
CLASS_LABEL=388     # ImageNet water ouzel — set to empty string to sample unconditionally
NUM_SAMPLES=16
OUTPUT="$REPO_DIR/samples_drift-label${CLASS_LABEL}-step${CKPT_STEP}__imagenet256.png"

# Build class-label flag
CLASS_FLAG=""
if [ -n "$CLASS_LABEL" ]; then
    CLASS_FLAG="--class_label $CLASS_LABEL"
fi

python "$SCRIPT_DIR/image_sample_with_drift_imagenet256.py" \
    --ckpt_dir          "$CKPT_DIR"     \
    --ckpt_step         "$CKPT_STEP"    \
    --num_samples       $NUM_SAMPLES    \
    --num_steps_64      400             \
    --num_steps_128     300             \
    --num_steps_256     300             \
    --seed              42              \
    --noise_schedule    linear          \
    --num_res_blocks    2               \
    --num_heads         4               \
    --num_channels_256  192             \
    --num_channels_128  128             \
    --num_channels_64   64              \
    --attention_resolutions_256 "32,16,8" \
    --attention_resolutions_128 "32,16,8" \
    --attention_resolutions_64  "16,8"    \
    --sample_scale      2               \
    --output            "$OUTPUT"       \
    $CLASS_FLAG

# DDIM variant (faster, deterministic):
# Add --use_ddim --ddim_eta 0.0 to the command above
