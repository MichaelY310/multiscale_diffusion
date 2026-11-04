#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=4
"$PY" -m unittest discover -s tests -p 'test_csra_*.py'
check_memory() {
  mapfile -t values < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
  [[ ${#values[@]} == 4 ]] || return 1
  for value in "${values[@]}"; do (( value >= 16000 )) || return 1; done
}
until check_memory; do echo 'Waiting for16GiB/card on0–3; unrelated jobs untouched'; sleep 30; done
pids=()
seeds=(20260921 20260921 20275004 20275004)
arms=(aligned zero aligned zero)
for gpu in 0 1 2 3; do
  seed="${seeds[$gpu]}"; arm="${arms[$gpu]}"
  out="eval_results/csra_prefix8_conditional_rollout_shift_20260913/seed_${seed}_${arm}.json"
  [[ ! -f "$out" ]] || continue
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/diagnose_csra_prefix_rollout_shift.py \
    --seed "$seed" --arm "$arm" > "logs/csra_prefix8_conditional_shift_${seed}_${arm}_20260913.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || exit 1
echo 'Paired teacher-state/student-state diagnostic complete'
