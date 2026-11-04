#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=2
"$PY" -m unittest discover -s tests -p 'test_csra_*.py'
while true; do
  mapfile -t free < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
  ready=1
  [[ ${#free[@]} == 4 ]] || ready=0
  for n in "${free[@]}"; do (( n >= 10000 )) || ready=0; done
  (( ready )) && break
  echo 'Waiting for 10,000MiB free per card, no other jobs will be stopped'
  sleep 30
done
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/probe_rdd_native_coarse_kv.py \
    --shard "$gpu" --output-dir "eval_results/csra_native_kv_20260913/shard_$gpu" \
    > "logs/csra_native_kv_${gpu}_20260913.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || exit 1
echo 'Native coarse KV diagnostic complete'
