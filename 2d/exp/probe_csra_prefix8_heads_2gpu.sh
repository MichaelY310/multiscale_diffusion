#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=4
"$PY" -m unittest discover -s tests -p 'test_csra_*.py'
check_memory() {
  mapfile -t values < <(nvidia-smi -i 0,1 --query-gpu=memory.free --format=csv,noheader,nounits)
  [[ ${#values[@]} == 2 ]] || return 1
  for value in "${values[@]}"; do (( value >= 18000 )) || return 1; done
}
until check_memory; do echo 'Waiting for18GiB on0/1; unrelated jobs untouched'; sleep 30; done
pids=();seeds=(20260921 20260923)
for gpu in 0 1; do
  seed="${seeds[$gpu]}"
  out="eval_results/csra_prefix8_conditional_head_oracle_20260913/seed_${seed}/metrics.json"
  [[ ! -f "$out" ]] || continue
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/probe_csra_prefix_output_heads.py \
    --seed "$seed" > "logs/csra_prefix8_heads_${seed}_20260913.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || exit 1
echo 'Two-seed mean/variance head oracle previews complete'
