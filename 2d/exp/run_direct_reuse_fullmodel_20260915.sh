#!/usr/bin/env bash
# Bounded adapter-free pilot. Independent full256 arms; GPU0-3 ONLY.
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
R=/home/yuheng/rdd_experiments/direct_reuse_fullmodel_20260915
mkdir -p "$R"
exec > >(tee -a "$R/launcher.log") 2>&1
exec 9>"$R/launcher.lock"
flock -n 9 || { echo 'Already running'; exit 1; }
export OMP_NUM_THREADS=2 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
PY="$PWD/.venv-rdd-eval/bin/python"
VY="$PWD/.venv-rdd-video/bin/python"
wait_memory() {
  while true; do
    local ok=1
    while IFS= read -r free; do (( free >= 32000 )) || ok=0; done < <(
      nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits)
    ((ok)) && break
    echo "Waiting for32,000MiB free on $1; not disturbing others"
    sleep 30
  done
}
"$PY" -m unittest discover -s tests -p test_direct_reuse_fullmodel.py
wait_memory 0,1,2,3
all_pids=()
bash 2d/exp/run_direct_reuse_training_gpu01_20260915.sh &
all_pids+=("$!")
for item in '2 832 29841' '3 1280 29842'; do
  read -r gpu size port <<< "$item"
  (
    if [[ ! -f "$R/video_$size/metrics.json" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" "$VY" -u rdd_video/benchmark_direct_reuse_fullmodel.py \
        --output "$R/video_$size" --size "$size" --port "$port" \
        > "$R/video_$size.log" 2>&1
    fi
    if [[ "$gpu" == 2 && ! -f "$R/video_temporal221/metrics.json" ]]; then
      CUDA_VISIBLE_DEVICES=2 "$VY" -u rdd_video/benchmark_direct_reuse_temporal221.py \
        --output "$R/video_temporal221" --size 832 --port 29843 --groups 4 \
        > "$R/video_temporal221.log" 2>&1
    fi
  ) &
  all_pids+=("$!")
done
failed=0
for pid in "${all_pids[@]}"; do wait "$pid" || failed=1; done
((failed == 0)) || exit 1
bash 2d/exp/run_direct_reuse_preview_20260915.sh
echo 'COMPLETE audited full-model pilot:64 paired images and3 video cases. Inspect quality before extending.'
