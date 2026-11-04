#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
R=/home/yuheng/rdd_experiments/reuse_latency_tiers_20260915
GPU=${1:?0 or1}
case "$GPU" in
 0) waitcase=temporal221_832_29; name=temporal421_1920_61; frames=61 ;;
 1) waitcase=temporal421_832_29; name=temporal421_1920_125; frames=125 ;;
 *) exit 2 ;;
esac
exec > >(tee -a "$R/large_gpu${GPU}_launcher.log") 2>&1
until [[ -f "$R/$waitcase/metrics.json" ]]; do sleep 10; done
exec 4>"$R/gpu${GPU}.lock"
flock 4
while true; do
  available=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
  ((available>=32000)) && break
  sleep 20
done
export CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv-rdd-video/bin/python -u rdd_video/benchmark_reuse_latency_tiers.py \
 --output "$R/$name" --port "$((30080+GPU))" --groups 3 --warmups 1 \
 --vae-precision bf16 --width 1920 --height 1024 --frames "$frames" --factors temporal421 \
 --variants fine24,cascade24,cascade28 > "$R/$name.log" 2>&1
