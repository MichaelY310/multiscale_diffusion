#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=2
"$PY" -m unittest discover -s tests -p 'test_csra_*.py'
while true; do
  mapfile -t free < <(nvidia-smi -i 0,1,2 --query-gpu=memory.free --format=csv,noheader,nounits)
  ready=1
  [[ ${#free[@]} == 3 ]] || ready=0
  for n in "${free[@]}"; do (( n >= 12000 )) || ready=0; done
  (( ready )) && break
  echo 'Waiting for12,000MiB/card on physical0,1,2; no process will be stopped'
  sleep 30
done
seeds=(20260921 20260923 20275004)
pids=()
for gpu in 0 1 2; do
  seed="${seeds[$gpu]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_rdd_native_kv_preview.py \
    --seed "$seed" --output-dir "eval_results/csra_native_kv_preview_20260913/seed_$seed" \
    > "logs/csra_native_kv_preview_${seed}_20260913.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed==0 )) || exit 1
echo 'Three paired native-KV preview seeds complete'
