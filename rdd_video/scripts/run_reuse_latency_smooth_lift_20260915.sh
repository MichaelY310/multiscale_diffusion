#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
R=/home/yuheng/rdd_experiments/reuse_latency_tiers_20260915
exec > >(tee -a "$R/smooth_lift_launcher.log") 2>&1
until [[ -f "$R/temporal221_832_29_person/metrics.json" ]]; do sleep 10; done
exec 4>"$R/gpu2.lock"
flock 4
while true; do
 free=$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits)
 ((free>=32000)) && break
 sleep 20
done
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv-rdd-video/bin/python -u rdd_video/benchmark_reuse_latency_smooth_lift.py \
 --output "$R/spatial221_1280_81_trilinear" --groups 3 \
 > "$R/spatial221_1280_81_trilinear.log" 2>&1
