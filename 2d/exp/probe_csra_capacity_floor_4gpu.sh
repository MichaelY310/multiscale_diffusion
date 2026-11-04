#!/usr/bin/env bash
set -euo pipefail
ROOT=/scr/yuheng/DevOpsBench/multiscale_diffusion
cd "$ROOT"
PYTHON="$ROOT/.venv-rdd-eval/bin/python"
OUT=eval_results/csra_capacity_floor_20260913
[[ ! -e "$OUT" ]] || exit 1
export OMP_NUM_THREADS=4
CUDA_VISIBLE_DEVICES='' "$PYTHON" -m unittest discover -s tests -p 'test_csra_*.py' -v
mapfile -t free_mib < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
[[ ${#free_mib[@]} == 4 ]] || exit 1
for free in "${free_mib[@]}"; do (( free >= 12000 )) || exit 1; done
mkdir -p "$OUT"
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u 2d/attention_analysis/probe_csra_capacity_floor.py \
    --base-ckpt checkpoints/RDD-Unified-L/0400000.pt \
    --adapter-ckpt checkpoints/RDD-CSRA-condition-v2-L400K-20260912/best.pt \
    --data-root eval_results/csra_rollout_data4k_20260912 --shard "$gpu" \
    --output-dir "$OUT/shard_$gpu" > "$OUT/shard_$gpu.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed==0 )) || exit 1
echo 'All four output-capacity-floor probes complete.'
