#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=2
exec 9>eval_results/csra_bridge_bands_20260914.lock
flock -n 9 || { echo 'Exact bridge-band audit already running'; exit 1; }
"$PY" -m unittest discover -s tests -p 'test_csra_*.py'
while true; do
  mapfile -t free < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
  ready=1; [[ ${#free[@]} == 4 ]] || ready=0
  for n in "${free[@]}"; do ((n>=20000)) || ready=0; done
  ((ready)) && break
  echo 'Waiting for20,000MiB free/card; other jobs untouched'; sleep 30
done
pids=()
for gpu in 0 1 2 3; do
  [[ ! -f "eval_results/csra_bridge_bands_20260914/shard_${gpu}.json" ]] || continue
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/audit_csra_bridge_bands.py \
    --shard "$gpu" > "logs/csra_bridge_bands_gpu${gpu}_20260914.log" 2>&1 &
  pids+=("$!")
done
failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed==0)) || exit 1
echo 'Four oracle cell-band audits complete. Review before any new training.'
