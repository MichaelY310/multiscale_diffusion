#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
ROOT=/home/yuheng/rdd_experiments/csra_skip_layout_20260915
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec 9>"$ROOT/attention_gate_pilot.lock"
flock -n 9 || exit 1
for name in split800 split1600 front800 front1600;do test -f "$ROOT/split_rollout_errors/$name/metrics.json";done
pids=()
for gpu in 0 1 2 3;do
 (
  while true;do
   mem=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
   if [[ $mem =~ ^[0-9]+$ ]] && ((mem>=22528));then break;fi
   echo "Waiting22GiB free on$gpu;4-7 reserved";sleep 30
  done
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_csra_attention_gate_pilot.py --root "$ROOT" --rank "$gpu"
 ) >> "$ROOT/attention_gate_pilot_gpu${gpu}.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}";do wait "$pid" || failed=1;done
((failed==0)) || { echo 'Exact gate-quality probe failed';exit 1; }
echo 'Readout gate pilot complete; no training and no runtime gain claimed.'
