#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
R=/home/yuheng/rdd_experiments/aggressive_direct_reuse_20260915
mkdir -p "$R"
exec > >(tee -a "$R/video_mild24_bf16_launcher.log") 2>&1
exec 4>"$R/video_mild24_bf16.lock"
flock -n 4 || exit 1
until [[ -f "$R/video_832_temporal221_bf16/metrics.json" ]]; do sleep 20; done
while true; do
  free=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits)
  ((free>=32000)) && break
  sleep 20
done
export OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=1 .venv-rdd-video/bin/python -u rdd_video/benchmark_aggressive_reuse_mild_bf16.py \
  --output "$R/video_832_temporal221_bf16_high24" --size 832 --port 29962 --groups 4 \
  --variants fine24 --factors temporal221 --vae-precision bf16 \
  > "$R/video_832_temporal221_bf16_high24.log" 2>&1
