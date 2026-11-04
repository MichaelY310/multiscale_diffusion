#!/usr/bin/env bash
# Bounded, small paired screen followed by a new layer-layout information probe.
# Does not signal any training job; uses only physical GPUs0-3, <=18% allocator/card.
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
ROOT=/home/yuheng/rdd_experiments/csra_skip_layout_20260915
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p "$ROOT"
exec 9>"$ROOT/run.lock"
flock -n 9 || { echo 'This exact exploration is already running'; exit 1; }
"$PY" -m unittest discover -s tests -p 'test_csra_skip_schedule.py'
check_memory() {
 local m
 mapfile -t mem < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
 [[ ${#mem[@]} == 4 ]] || return 1
 for m in "${mem[@]}"; do ((m >= 22528)) || return 1; done
}
until check_memory; do echo 'Waiting for22GiB free on0-3; other jobs untouched'; sleep 30; done
pids=()
for gpu in 0 1 2 3; do
 CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_csra_skip_schedules.py sample \
  --root "$ROOT" --rank "$gpu" >> "$ROOT/schedules_gpu${gpu}.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed==0)) || { echo 'Schedule screen failed: inspect only this run logs'; exit 1; }
"$PY" 2d/attention_analysis/sample_csra_skip_schedules.py summarize --root "$ROOT"
echo 'Schedule screen complete. Layer probe is a separate bounded follow-up, never a video speed claim.'
