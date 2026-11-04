#!/usr/bin/env bash
set -euo pipefail
ROOT=/scr/yuheng/DevOpsBench/multiscale_diffusion
cd "$ROOT"
GPU=${GPU:-0}
case "$GPU" in 0|1|2|3) ;; *) echo 'Only one physical GPU0-3 permitted'; exit 2;; esac
OUT=eval_results/csra_neighbor_video_feasibility_20260914
mkdir -p "$OUT"
exec 9>"$OUT/run.lock"
flock -n 9 || { echo 'This exact feasibility job already running'; exit 0; }
if [[ -f "$OUT/full/metrics.json" ]]; then echo 'Already complete'; exit 0; fi
export CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
PY=.venv-rdd-video/bin/python
"$PY" -m unittest discover -s tests -p test_csra_prefix_video_bridge.py -v 2>&1 | tee "$OUT/existing_tests.log"
"$PY" -m unittest discover -s tests -p test_csra_fine_neighbor.py -v 2>&1 | tee "$OUT/neighbor_tests.log"
headroom() {
    while true; do
        free=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
        if [[ "$free" =~ ^[0-9]+$ ]] && (( free >= 32768 )); then break; fi
        echo "GPU$GPU free=$free MiB; waiting for32GiB. No process will be signaled."
        sleep 45
    done
}
headroom
if [[ ! -f "$OUT/smoke/metrics.json" ]]; then
    "$PY" rdd_video/benchmark_csra_prefix_feasibility.py --output "$OUT/smoke" \
        --front-keep 1 --neighbor-rank 128 --smoke --groups 1 --warmups 1 --port 29714 \
        2>&1 | tee "$OUT/smoke.log"
fi
headroom
"$PY" rdd_video/benchmark_csra_prefix_feasibility.py --output "$OUT/full" \
    --front-keep 1 --neighbor-rank 128 --groups 12 --warmups 2 --port 29714 \
    2>&1 | tee "$OUT/full.log"
