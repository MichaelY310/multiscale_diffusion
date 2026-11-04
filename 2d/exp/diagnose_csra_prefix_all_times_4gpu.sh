#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=4
exec 9>eval_results/csra_prefix8_all_times_20260913.lock
flock -n 9 || { echo 'Exact all-times probe already running'; exit 1; }
"$PY" -m unittest discover -s tests -p 'test_csra_*.py'
check_memory() {
  mapfile -t values < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
  [[ ${#values[@]} == 4 ]] || return 1
  for value in "${values[@]}"; do (( value >= 18000 )) || return 1; done
}
until check_memory; do echo 'Waiting for18GiB on0–3; unrelated jobs untouched'; sleep 30; done
pids=();budgets=(1600 1600 3200 3200);arms=(aligned zero aligned zero)
for gpu in 0 1 2 3; do
  (
    for seed in 20260921 20275004; do
      result="eval_results/csra_prefix8_all_times_20260913/budget_${budgets[$gpu]}_${arms[$gpu]}/seed_${seed}.json"
      [[ ! -f "$result" ]] || continue
      CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/diagnose_csra_prefix_all_times.py \
        --budget "${budgets[$gpu]}" --arm "${arms[$gpu]}" --seed "$seed"
    done
  ) > "logs/csra_prefix8_all_times_${budgets[$gpu]}_${arms[$gpu]}_20260913.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || exit 1
CUDA_VISIBLE_DEVICES='' "$PY" 2d/attention_analysis/summarize_csra_prefix_all_times.py
echo 'All90-time diagnostic complete:4budget/source combinations x2seeds'
