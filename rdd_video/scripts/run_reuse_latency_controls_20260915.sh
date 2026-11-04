#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
R=/home/yuheng/rdd_experiments/reuse_latency_tiers_20260915
GPU=${1:?0 or1}
case "$GPU" in 0|1) ;; *) exit 2;; esac
exec > >(tee -a "$R/controls_gpu${GPU}_launcher.log") 2>&1
exec 4>"$R/gpu${GPU}.lock"
flock -n 4 || exit 1
available=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
((available>=32000)) || exit 3
export CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ "$GPU" == 0 ]]; then
 .venv-rdd-video/bin/python -u rdd_video/benchmark_reuse_latency_tiers_controls.py \
  --output "$R/temporal221_832_29_head_controls" --port 30090 --groups 4 \
  > "$R/temporal221_832_29_head_controls.log" 2>&1
else
 .venv-rdd-video/bin/python -u rdd_video/benchmark_reuse_latency_tiers.py \
  --output "$R/temporal421_1600_61" --port 30091 --groups 4 --warmups 1 \
  --width 1600 --height 896 --frames 61 --factors temporal421 \
  --variants fine24,cascade24,cascade28 > "$R/temporal421_1600_61.log" 2>&1
fi
