#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
R=/home/yuheng/rdd_experiments/direct_reuse_fullmodel_20260915
exec > >(tee -a "$R/video_temporal221.log") 2>&1
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 .venv-rdd-video/bin/python -u \
  rdd_video/benchmark_direct_reuse_temporal221.py --output "$R/video_temporal221" \
  --size 832 --port 29844 --groups 4
