#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
export OMP_NUM_THREADS=4
pids=()
for gpu in 0 1 2 3; do
  (
    while true; do
      free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
      if (( free >= 12000 )); then break; fi
      echo "GPU $gpu waiting for 12000 MiB free; no process will be stopped."
      sleep 30
    done
    CUDA_VISIBLE_DEVICES="$gpu" .venv-rdd-eval/bin/python -u \
      2d/attention_analysis/probe_csra_preserved_shift.py --shard "$gpu" --max-files 1 \
      --output-dir "eval_results/csra_preserved_shift_20260913/shard_$gpu"
  ) > "logs/csra_preserved_shift_gpu${gpu}_20260913.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
