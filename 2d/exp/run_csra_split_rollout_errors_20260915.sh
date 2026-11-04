#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
ROOT=/home/yuheng/rdd_experiments/csra_skip_layout_20260915
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec 9>"$ROOT/split_rollout_errors.lock"
flock -n 9 || exit 1
test -f "$ROOT/split1600_quality256/summary_and_audit.json"
arms=(split800 split1600 front800 front1600)
pids=()
for gpu in 0 1 2 3; do
 (
  while true; do
   mem=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
   if [[ $mem =~ ^[0-9]+$ ]] && ((mem>=22528)); then break; fi
   echo "Waiting22GiB free on$gpu";sleep 30
  done
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/probe_csra_split_rollout_errors.py \
   --root "$ROOT" --arm "${arms[$gpu]}"
 ) >> "$ROOT/split_rollout_errors_${arms[$gpu]}.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}";do wait "$pid" || failed=1;done
((failed==0)) || { echo 'Exact diagnostic failed; inspect its log'; exit 1; }
echo 'Read-only error/conditional-source probe complete; no training extension.'
