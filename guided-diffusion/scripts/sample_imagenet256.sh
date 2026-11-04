#!/bin/bash
# Sample from the standard (no-drift) class-conditional 256x256 UNet.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

cd "$REPO_DIR"
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3         # single GPU is enough for sampling
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Checkpoint to sample from ----
CKPT_DIR="/scr/yuheng/DevOpsBench/multiscale_diffusion/guided-diffusion/results_imagenet256/20260324-045110-imagenet256-standard"
CKPT_STEP=0115000   # 7-digit step, e.g. 0015000

# ---- Output ----
CLASS_LABEL=388     # ImageNet class index; leave empty for random labels
NUM_SAMPLES=16
OUTPUT="$REPO_DIR/samples_original-label${CLASS_LABEL}-step${CKPT_STEP}__imagenet256.png"

# Build class-label flag
CLASS_FLAG=""
if [ -n "$CLASS_LABEL" ]; then
    CLASS_FLAG="--class_label $CLASS_LABEL"
fi

python "$SCRIPT_DIR/image_sample_imagenet256.py" \
    --ckpt          "${CKPT_DIR}/checkpoints/${CKPT_STEP}.pt" \
    --num_samples   $NUM_SAMPLES    \
    --num_steps     250             \
    --ddim_eta      0.0             \
    --seed          42              \
    --output        "$OUTPUT"       \
    $CLASS_FLAG

# DDIM variant (faster, deterministic):
# Add --use_ddpm for ancestral DDPM sampling (1000 steps)
