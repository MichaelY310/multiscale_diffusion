#!/usr/bin/env bash
# Only physical0-3, original FastWan code/weights unchanged.
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
ROOT=/home/yuheng/rdd_experiments/csra_skip_layout_20260915
PY=.venv-rdd-video/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
exec 9>"$ROOT/split_video_cost.lock"
flock -n 9 || { echo 'Exact video cost job already running'; exit 1; }
test -f "$ROOT/split_quality256/summary_and_audit.json"
CUDA_VISIBLE_DEVICES='' "$PY" -m unittest discover -s tests -p test_csra_split_video_bridge.py -v
cases=(832_spatial 832_st 1280_spatial 1280_st)
pids=()
for gpu in 0 1 2 3; do
 case=${cases[$gpu]}
 if [[ -f "$ROOT/split_video_cost/$case/metrics.json" ]]; then continue; fi
 (
  while true; do
   mem=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
   if [[ $mem =~ ^[0-9]+$ ]] && ((mem>=73728)); then break; fi
   echo "Waiting72GiB on physical$gpu; other jobs untouched"; sleep 30
  done
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u rdd_video/benchmark_csra_split_feasibility.py \
   --root "$PWD" --output "$ROOT/split_video_cost/$case" --case "$case" \
   --port "$((29760+gpu))" --groups 8 --warmups 2
 ) >> "$ROOT/split_video_cost_${case}.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed==0)) || { echo 'A named video worker failed; inspect its exact log'; exit 1; }
echo 'Complete two-reader video feasibility; training/quality NOT claimed.'
