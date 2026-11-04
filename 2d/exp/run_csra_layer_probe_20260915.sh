#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
ROOT=/home/yuheng/rdd_experiments/csra_skip_layout_20260915
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$ROOT"
exec 9>"$ROOT/layer_probe.lock"
flock -n 9 || { echo 'This exact layer probe already runs'; exit 1; }
while [[ ! -f "$ROOT/schedules/summary.json" ]]; do
 if ! tmux has-session -t =csra_skip_layout_20260915 2>/dev/null; then
  echo 'Sampling prerequisite failed or stopped. Do not silently proceed.'; exit 1
 fi
 sleep 30
done
check_memory() {
 local m
 mapfile -t mem < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
 [[ ${#mem[@]} == 4 ]] || return 1
 for m in "${mem[@]}"; do ((m>=18432)) || return 1; done
}
until check_memory; do echo 'Waiting18GiB free on0-3; other jobs untouched'; sleep 30; done
layouts=(front0_6 front3_6 front6_6 split3_3)
for phase in smoke full; do
 pids=(); extra=()
 [[ "$phase" != smoke ]] || extra=(--smoke)
 for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/probe_csra_layer_layouts.py \
   --layout "${layouts[$gpu]}" --root "$ROOT" "${extra[@]}" >> "$ROOT/probe_${phase}_${layouts[$gpu]}.log" 2>&1 &
  pids+=("$!")
 done
 failed=0
 for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
 ((failed==0)) || { echo 'Layout probe failed. Inspect only this run logs'; exit 1; }
done
echo 'All four matched six-block probes completed; pending quality/cost-based training follow-up.'
