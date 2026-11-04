#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
ROOT=/home/yuheng/rdd_experiments/csra_skip_layout_20260915
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec 9>"$ROOT/aligned_pointwise256.lock"
flock -n 9 || exit 1
CUDA_VISIBLE_DEVICES='' "$PY" -m unittest discover -s tests -p test_csra_aligned_pointwise_reader.py -v
pids=()
for gpu in 0 1 2 3;do
 (
  while true;do
   mem=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
   if [[ $mem =~ ^[0-9]+$ ]] && ((mem>=22528));then break;fi
   echo "Waiting22GiB free on$gpu;4-7 reserved";sleep 30
  done
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_csra_aligned_pointwise256.py --root "$ROOT" --rank "$gpu"
 ) >> "$ROOT/aligned_pointwise256_gpu${gpu}.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}";do wait "$pid" || failed=1;done
((failed==0)) || { echo 'Actual bypass pilot failed';exit 1; }
echo 'Actual pointwise256 quality complete; no training or speed claim.'
