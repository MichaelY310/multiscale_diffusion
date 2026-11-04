#!/usr/bin/env bash
# Additional matched deployment setting, NOT compared across VAE precision.
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
R=/home/yuheng/rdd_experiments/aggressive_direct_reuse_20260915
mkdir -p "$R"
exec > >(tee -a "$R/video_mild_bf16_launcher.log") 2>&1
exec 5>"$R/video_mild_bf16.lock"
flock -n 5 || exit 1
until [[ -f "$R/train_reuse18/complete.json" ]]; do
  echo 'Waiting for our GPU1 image training'; sleep 30
done
while true; do
  free=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits)
  ((free>=32000)) && break
  echo 'Waiting for32,000MiB free on GPU1'; sleep 30
done
export OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=1 .venv-rdd-video/bin/python -u rdd_video/benchmark_aggressive_reuse_mild_bf16.py \
  --output "$R/video_832_temporal221_bf16" --size 832 --port 29961 --groups 4 \
  --variants fine18 --factors temporal221 --vae-precision bf16 \
  > "$R/video_832_temporal221_bf16.log" 2>&1
