#!/bin/bash
# Train a standard (no-drift) class-conditional 256×256 UNet on ImageNet.
# Single model, full 1000-step DDPM schedule.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

source "$REPO_DIR/.venv/bin/activate"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=WARN
export HF_DATASETS_CACHE=/scr/yuheng/DevOpsBench/multiscale_diffusion/.hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RESULTS_DIR="$REPO_DIR/results_imagenet256"

# ---- Resume: set to checkpoint path to resume (leave empty for fresh start) ----
RESUME_CKPT=

mkdir -p "$RESULTS_DIR"

MPIEXEC=/usr/mpi/gcc/openmpi-4.1.5rc2/bin/mpiexec

cd "$REPO_DIR"

$MPIEXEC -n 4 -x CUDA_VISIBLE_DEVICES \
    python "$SCRIPT_DIR/image_train_imagenet256.py" \
        --results_dir               "$RESULTS_DIR"  \
        --image_size                256             \
        --num_channels              192             \
        --num_res_blocks            2               \
        --num_heads                 4               \
        --attention_resolutions     "32,16,8"       \
        --noise_schedule            linear          \
        --diffusion_steps           1000            \
        --batch_size                48              \
        --num_workers               4               \
        --lr                        1e-4            \
        --weight_decay              0.0             \
        --ema_rate                  0.9999          \
        --total_steps               1000000         \
        --log_every                 100             \
        --ckpt_every                5000            \
        --use_checkpoint            \
        ${RESUME_CKPT:+--resume "$RESUME_CKPT"}
