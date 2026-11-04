#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=2
exec 9>eval_results/csra_prefix8_bypass_20260913.lock
flock -n 9 || { echo 'This exact diagnostic is already running'; exit 1; }
"$PY" -m unittest discover -s tests -p 'test_csra_*.py'
check_memory() {
  mapfile -t values < <(nvidia-smi -i 0,1,2 --query-gpu=memory.free --format=csv,noheader,nounits)
  [[ ${#values[@]} == 3 ]] || return 1
  for value in "${values[@]}"; do (( value >= 22000 )) || return 1; done
}
until check_memory; do echo 'Waiting for22GiB/card on0–2; other jobs untouched'; sleep 30; done
pids=();seeds=(20260921 20260923 20275004)
for gpu in 0 1 2; do
  seed="${seeds[$gpu]}"
  [[ ! -f "eval_results/csra_prefix8_bypass_20260913/seed_${seed}/metrics.json" ]] || continue
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/probe_csra_prefix_bypass.py --seed "$seed" \
    > "logs/csra_prefix8_bypass_${seed}_20260913.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || exit 1
echo 'All3diagnostic seeds complete; this is NOT an accelerated variant'
