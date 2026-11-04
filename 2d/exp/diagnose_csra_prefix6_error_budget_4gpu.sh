#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=2
exec 9>eval_results/csra_prefix6_error_budget_20260914.lock
flock -n 9 || exit 1
"$PY" -m unittest discover -s tests -p 'test_csra_*.py'
check_memory() {
 mapfile -t mem < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
 [[ ${#mem[@]} == 4 ]] || return 1
 for m in "${mem[@]}"; do ((m>=22528)) || return 1; done
}
until check_memory; do echo 'Waiting for22GiB free/card0-3; other tasks untouched'; sleep 30; done
seeds=(20660914 20660914 20670914 20670914)
arms=(aligned zero aligned zero)
pids=()
for gpu in 0 1 2 3; do
 [[ ! -f "eval_results/csra_prefix6_error_budget_20260914/seed_${seeds[$gpu]}_${arms[$gpu]}.json" ]] || continue
 CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/diagnose_csra_prefix6_error_budget.py \
  --root eval_results/csra_prefix6_error_budget_20260914 --seed "${seeds[$gpu]}" --arm "${arms[$gpu]}" --batch 16 \
  > "logs/csra_prefix6_error_budget_gpu${gpu}_20260914.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed==0)) || exit 1
echo 'Four paired all90-time probes completed; analyze before selecting training changes.'
