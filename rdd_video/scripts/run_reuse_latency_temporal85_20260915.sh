#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
R=/home/yuheng/rdd_experiments/reuse_latency_tiers_20260915
exec > >(tee -a "$R/temporal85_launcher.log") 2>&1
until [[ -f "$R/temporal421_1280_61/metrics.json" ]]; do sleep 10; done
exec 4>"$R/gpu3.lock"
flock 4
while true; do
 free=$(nvidia-smi -i 3 --query-gpu=memory.free --format=csv,noheader,nounits)
 ((free>=32000)) && break
 sleep 20
done
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv-rdd-video/bin/python -u rdd_video/benchmark_reuse_latency_tiers.py \
 --output "$R/temporal221_1280_85" --port 30113 --groups 4 --warmups 1 \
 --width 1280 --height 768 --frames 85 --factors temporal221 \
 --variants fine24,fine28 > "$R/temporal221_1280_85.log" 2>&1
