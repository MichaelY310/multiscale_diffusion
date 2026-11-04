#!/usr/bin/env bash
set -euo pipefail
ROOT=/scr/yuheng/DevOpsBench/multiscale_diffusion
cd "$ROOT"
GPU=${GPU:-0}
case "$GPU" in 0|1|2|3) ;; *) echo 'Only one physical GPU 0–3 permitted'; exit 2;; esac
OUT=eval_results/csra_prefix_video_feasibility_20260913
mkdir -p "$OUT" logs
exec 9>"$OUT/run.lock"
flock -n 9 || { echo 'This exact feasibility job is already running'; exit 0; }
if [[ -f "$OUT/full/metrics.json" ]]; then
    echo 'Completed full feasibility metrics already exist; refusing duplicate run'
    exit 0
fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="$GPU"
PY=.venv-rdd-video/bin/python
"$PY" -m unittest discover -s tests -p test_csra_prefix_video_bridge.py -v
wait_headroom() {
    while true; do
        free=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
        if [[ "$free" =~ ^[0-9]+$ ]] && (( free >= 32768 )); then break; fi
        echo "Waiting for >=32GiB free on GPU$GPU; free=${free}MiB. No other process will be signaled."
        sleep 45
    done
}
wait_headroom
if [[ ! -f "$OUT/smoke/metrics.json" ]]; then
    "$PY" rdd_video/benchmark_csra_prefix_feasibility.py --output "$OUT/smoke" \
        --smoke --groups 1 --warmups 1 --port 29693 2>&1 | tee logs/csra_prefix_video_smoke_20260913.log
fi
wait_headroom
"$PY" rdd_video/benchmark_csra_prefix_feasibility.py --output "$OUT/full" \
    --groups 12 --warmups 2 --port 29693 2>&1 | tee logs/csra_prefix_video_full_20260913.log
