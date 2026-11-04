#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=2
"$PY" -m unittest discover -s tests -p 'test_csra_*.py'
arms=(v2_aligned v2_zero v5_aligned v5_zero)
pids=()
for gpu in 0 1 2 3; do
  (
    while true; do
      free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
      (( free >= 10000 )) && break
      echo "Waiting for 10,000MiB on GPU$gpu"
      sleep 30
    done
    arm="${arms[$gpu]}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/probe_csra_time_source_advantage.py \
      --arm "$arm" --output-dir "eval_results/csra_time_source_20260913/$arm" \
      2>&1 | tee "logs/csra_time_source_${arm}_20260913.log"
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || exit 1
echo 'Matched trained-source time/band diagnostics complete.'
