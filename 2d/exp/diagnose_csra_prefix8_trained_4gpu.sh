#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=4
out=eval_results/csra_prefix8_trained_diagnostic_20260913
mkdir -p "$out" logs
"$PY" -m unittest discover -s tests -p 'test_csra_*.py'
check_memory() {
  mapfile -t values < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
  [[ ${#values[@]} == 4 ]] || return 1
  for value in "${values[@]}"; do (( value >= 16000 )) || return 1; done
}
until check_memory; do echo 'Waiting for16GiB/card on0–3; unrelated processes untouched'; sleep 30; done
pids=()
for gpu in 0 1 2 3; do
  [[ ! -f "$out/shard_${gpu}.json" ]] || continue
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/diagnose_csra_trained_prefix.py run \
    --shard "$gpu" --output-root "$out" > "logs/csra_prefix8_diagnostic_${gpu}_20260913.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || exit 1
"$PY" 2d/attention_analysis/diagnose_csra_trained_prefix.py aggregate --output-root "$out"
