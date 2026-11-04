#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
R=/home/yuheng/rdd_experiments/aggressive_direct_reuse_20260915
mkdir -p "$R"
exec > >(tee -a "$R/video_long_launcher.log") 2>&1
exec 6>"$R/video_long.lock"
flock -n 6 || exit 1
PY="$PWD/.venv-rdd-video/bin/python"
export OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pids=()
for item in '2 cascade24 fine 29951' '3 fine24 cascade 29952'; do
  read -r gpu variant prerequisite port <<< "$item"
  (
    until [[ -f "$R/video_1280_$prerequisite/metrics.json" ]]; do
      echo "Waiting for GPU$gpu previous owned video test"; sleep 30
    done
    while true; do
      free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
      ((free>=32000)) && break
      echo "Waiting for32,000MiB free on $gpu"; sleep 30
    done
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u rdd_video/benchmark_aggressive_direct_reuse_long.py \
      --output "$R/video_1280_81f_$variant" --size 1280 --frames 81 --port "$port" --groups 3 \
      --variants "$variant" --factors spatial421 --vae-precision fp32 \
      > "$R/video_1280_81f_$variant.log" 2>&1
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed==0)) || exit 1
echo 'COMPLETE matched81-frame FP32 VAE tests'
