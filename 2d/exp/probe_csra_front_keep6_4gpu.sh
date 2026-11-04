#!/usr/bin/env bash
# Read-only teacher capture + two matched placement probes; no original training
# or sampler changes. Only physical0-3; never signal any other process.
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=4
mkdir -p logs
"$PY" -m unittest discover -s tests -p 'test_csra_front_keep.py'
"$PY" -m unittest discover -s tests -p 'test_csra_prefix_probe.py'
while true; do
  mapfile -t free < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
  ready=1
  [[ ${#free[@]} == 4 ]] || ready=0
  for n in "${free[@]}"; do (( n >= 13000 )) || ready=0; done
  (( ready )) && break
  echo 'Waiting for13,000MiB free on0-3; leaving existing jobs untouched'
  sleep 30
done
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/probe_csra_front_keep.py capture --shard "$gpu" \
    > "logs/csra_front_keep6_capture_gpu${gpu}_20260914.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || exit 1
pids=()
for front in 0 1; do
  CUDA_VISIBLE_DEVICES="$front" "$PY" -u 2d/attention_analysis/probe_csra_front_keep.py fit --front-keep "$front" \
    > "logs/csra_front_keep6_fit_front${front}_20260914.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || exit 1
echo 'Matched front0/front1 information and same-output diagnostic complete; review before training.'
