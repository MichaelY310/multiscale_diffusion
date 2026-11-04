#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
R=/home/yuheng/rdd_experiments/aggressive_direct_reuse_20260915
mkdir -p "$R"
exec > >(tee -a "$R/training_launcher.log") 2>&1
exec 8>"$R/train.lock"
flock -n 8 || exit 1
export OMP_NUM_THREADS=2
PY="$PWD/.venv-rdd-eval/bin/python"
pids=()
for item in '0 reuse12' '1 reuse18'; do
  read -r gpu mask <<< "$item"
  [[ ! -f "$R/train_$mask/complete.json" ]] || continue
  free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
  ((free>=32000)) || { echo "Insufficient memory on GPU$gpu; no eviction"; exit 1; }
  (
    if [[ ! -f "$R/smoke_$mask/complete.json" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/train_aggressive_direct_reuse.py \
        --output "$R/smoke_$mask" --mask "$mask" --steps 2 --per-gpu 256 \
        --validate-every 2 --smoke > "$R/smoke_$mask.log" 2>&1
    fi
    resume=()
    [[ ! -f "$R/train_$mask/latest.pt" ]] || resume=(--resume)
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/train_aggressive_direct_reuse.py \
      --output "$R/train_$mask" --mask "$mask" --steps 800 --per-gpu 256 \
      --validate-every 200 "${resume[@]}" > "$R/train_$mask.log" 2>&1
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed==0)) || exit 1
echo 'COMPLETE both aggressive adapter-free full-model finetunes'
