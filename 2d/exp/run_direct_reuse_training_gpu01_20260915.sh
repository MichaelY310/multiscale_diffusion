#!/usr/bin/env bash
# Parallel with video on2/3. Independent full256 arms avoid failing host NCCL.
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
R=/home/yuheng/rdd_experiments/direct_reuse_fullmodel_20260915
mkdir -p "$R"
exec > >(tee -a "$R/training_gpu01_launcher.log") 2>&1
exec 8>"$R/train.lock"
flock -n 8 || { echo 'Another training queue holds lease'; exit 1; }
export OMP_NUM_THREADS=2 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
PY="$PWD/.venv-rdd-eval/bin/python"
for gpu in 0 1; do
  free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
  ((free >= 32000)) || { echo "Insufficient free memory on $gpu"; exit 1; }
done
if [[ ! -f "$R/smoke256/complete.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$PY" -u 2d/attention_analysis/train_direct_reuse_fullmodel.py \
    --output "$R/smoke256" --steps 2 --per-gpu 256 --validate-every 2 --smoke \
    2>&1 | tee "$R/smoke256.log"
fi
gpu=0
pids=()
for arm in reuse6 dense; do
  [[ ! -f "$R/train_$arm/complete.json" ]] || continue
  resume=()
  [[ ! -f "$R/train_$arm/latest.pt" ]] || resume=(--resume)
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/train_direct_reuse_fullmodel.py \
    --output "$R/train_$arm" --mask "$arm" --steps 200 --per-gpu 256 --validate-every 100 "${resume[@]}" \
    >> "$R/train_$arm.log" 2>&1 &
  pids+=("$!")
  gpu=1
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed == 0)) || exit 1
echo 'Both global256 full-model arms complete; parent queue can generate previews after video finishes.'
