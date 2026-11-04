#!/usr/bin/env bash
# Only GPU2/3, never stop another process. Actual FP32 VAE remains identical.
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
R=/home/yuheng/rdd_experiments/aggressive_direct_reuse_20260915
mkdir -p "$R"
exec > >(tee -a "$R/video_launcher.log") 2>&1
exec 9>"$R/video.lock"
flock -n 9 || exit 1
PY="$PWD/.venv-rdd-video/bin/python"
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for gpu in 2 3; do
  free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
  ((free >= 32000)) || { echo "Insufficient free memory on $gpu; no eviction"; exit 1; }
done
CUDA_VISIBLE_DEVICES=2 "$PY" -u rdd_video/benchmark_aggressive_direct_reuse.py \
  --output "$R/video_1280_fine" --size 1280 --port 29941 --groups 4 \
  --variants fine18,fine24 --factors spatial421 --vae-precision fp32 \
  > "$R/video_1280_fine.log" 2>&1 &
p0=$!
CUDA_VISIBLE_DEVICES=3 "$PY" -u rdd_video/benchmark_aggressive_direct_reuse.py \
  --output "$R/video_1280_cascade" --size 1280 --port 29942 --groups 4 \
  --variants cascade24,cascade28 --factors spatial421 --vae-precision fp32 \
  > "$R/video_1280_cascade.log" 2>&1 &
p1=$!
failed=0
wait "$p0" || failed=1
wait "$p1" || failed=1
((failed==0)) || exit 1
echo 'COMPLETE aggressive matched FP32 VAE video screen'
