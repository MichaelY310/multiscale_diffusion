#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
R=/home/yuheng/rdd_experiments/reuse_latency_tiers_20260915
mkdir -p "$R"
GPU=${1:?physical GPU0-3}
case "$GPU" in 0|1|2|3) ;; *) exit 2;; esac
exec > >(tee -a "$R/launcher_gpu${GPU}.log") 2>&1
exec 4>"$R/gpu${GPU}.lock"
flock -n 4 || exit 1
export CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while true; do
  available=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
  ((available>=32000)) && break
  sleep 20
done
run_case() {
  name=$1
  shift
  .venv-rdd-video/bin/python -u rdd_video/benchmark_reuse_latency_tiers.py \
    --output "$R/$name" --port "$((30070+GPU))" --groups 3 --warmups 1 \
    --vae-precision bf16 "$@" > "$R/$name.log" 2>&1
}
case "$GPU" in
0) run_case temporal221_832_29 --width 832 --height 448 --frames 29 --factors temporal221 \
   --variants fine24,fine28,prefix18,prefix24,fine24_s075,prefix24_s075,fine24_s050 ;;
1) run_case temporal421_832_29 --width 832 --height 448 --frames 29 --factors temporal421 \
   --variants fine24,cascade12_24,cascade18_24,cascade24,cascade_prefix18_24,cascade28 ;;
2) run_case spatial221_1280_81 --width 1280 --height 768 --frames 81 --factors spatial221 \
   --variants fine18,fine24,prefix24,fine28,fine24_s075 ;;
3) run_case temporal421_1280_61 --width 1280 --height 768 --frames 61 --factors temporal421 \
   --variants fine24,cascade12_24,cascade18_24,cascade24,cascade_prefix18_24,cascade28 ;;
esac
printf 'COMPLETE GPU %s\n' "$GPU"
