#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
export OMP_NUM_THREADS=4
PY=.venv-rdd-eval/bin/python
"$PY" -m unittest discover -s tests -p 'test_csra_stage_isolation.py'
seeds=(20260921 20275003 20275004)
pids=()
for gpu in 0 1 2; do
  (
    while true; do
      free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
      if (( free >= 16000 )); then break; fi
      echo "GPU $gpu waiting for 16000 MiB; no other process will be stopped."
      sleep 30
    done
    seed=${seeds[$gpu]}
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" 2d/attention_analysis/probe_csra_joint_stage_isolation.py \
      --seed "$seed" --output-dir "eval_results/csra_joint_stage_isolation_20260913/seed_$seed"
  ) > "logs/csra_joint_stage_isolation_gpu${gpu}_20260913.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
exit "$status"
