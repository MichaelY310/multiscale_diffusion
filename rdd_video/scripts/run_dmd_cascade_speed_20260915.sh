#!/usr/bin/env bash
set -Eeuo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
ROOT=/home/yuheng/rdd_experiments/dmd_cascade_reuse_20260915
GPU=${1:-2}
FACTOR=${2:-spatial421}
NAME=${3:-video_spatial_speed}
VARIANTS=${4:-high_heavy,balanced}
mkdir -p "$ROOT/$NAME"
exec > >(tee -a "$ROOT/$NAME/launcher.log") 2>&1
mkdir -p "$ROOT/locks"
exec 9>"$ROOT/locks/gpu${GPU}.lock"
flock 9
export CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec .venv-rdd-video/bin/python -u rdd_video/benchmark_dmd_cascade_candidates.py \
 --output "$ROOT/$NAME" --groups 4 --port "$((30263+GPU))" --factors "$FACTOR" --variants "$VARIANTS"
