#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=2
exec 9>eval_results/csra_front_reader_audit_20260914.lock
flock -n 9 || { echo 'Exact reader audit already running';exit 1; }
"$PY" -m unittest discover -s tests -p 'test_csra_*.py'
for front in 0 1; do
  while true; do
    mapfile -t free < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
    ready=1;[[ ${#free[@]} == 4 ]] || ready=0
    for n in "${free[@]}";do ((n>=20000)) || ready=0;done
    ((ready)) && break
    echo 'Waiting for20,000MiB free/card; other jobs untouched';sleep 30
  done
  pids=()
  for gpu in 0 1 2 3;do
    output="eval_results/csra_front_reader_audit_20260914/front${front}/shard_${gpu}.json"
    [[ ! -f "$output" ]] || continue
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/audit_csra_front_reader.py \
      --front "$front" --shard "$gpu" \
      > "logs/csra_front_reader_front${front}_gpu${gpu}_20260914.log" 2>&1 &
    pids+=("$!")
  done
  failed=0;for pid in "${pids[@]}";do wait "$pid" || failed=1;done
  ((failed==0)) || exit 1
done
echo 'Eight read-only reader audits complete; no training or generation claims.'
