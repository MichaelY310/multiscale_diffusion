#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
R=/home/yuheng/rdd_experiments/reuse_latency_tiers_20260915
GPU=${1:?0 or2}
case "$GPU" in
0) dependency=temporal221_832_29_head_controls ;;
2) dependency=spatial221_1280_81 ;;
*) exit 2;;
esac
exec > >(tee -a "$R/followups_gpu${GPU}_launcher.log") 2>&1
until [[ -f "$R/$dependency/metrics.json" ]]; do sleep 10; done
exec 4>"$R/gpu${GPU}.lock"
flock 4
while true; do
 free=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
 ((free>=32000)) && break
 sleep 20
done
export CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ "$GPU" == 0 ]]; then
 .venv-rdd-video/bin/python -u rdd_video/benchmark_reuse_latency_tiers.py \
  --output "$R/temporal421_832_253" --port 30100 --groups 3 --warmups 1 \
  --width 832 --height 448 --frames 253 --factors temporal421 \
  --variants fine24,cascade24,cascade28 > "$R/temporal421_832_253.log" 2>&1
else
 for subject in car person; do
  if [[ "$subject" == car ]]; then
   prompt='A red sports car driving along a coastal road, ocean waves and cliffs, realistic cinematic tracking shot.'
   seed=20660914
  else
   prompt='A person wearing a yellow raincoat walking on a city street in light rain, realistic cinematic camera.'
   seed=20660915
  fi
  .venv-rdd-video/bin/python -u rdd_video/benchmark_reuse_latency_tiers_controls.py \
   --output "$R/temporal221_832_29_${subject}" --port 30102 --groups 4 --warmups 1 \
   --variants fine24,fine24_s075,fine28 --prompt "$prompt" --seed "$seed" \
   > "$R/temporal221_832_29_${subject}.log" 2>&1
 done
fi
