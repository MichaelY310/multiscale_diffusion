#!/usr/bin/env bash
set -Eeuo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
ROOT=/home/yuheng/rdd_experiments/dmd_cascade_reuse_20260915
exec > >(tee -a "$ROOT/monitor.log") 2>&1
exec .venv-rdd-eval/bin/python -u 2d/attention_analysis/monitor_dmd_cascade_jobs.py \
 --root "$ROOT" --families "${1:-high_heavy,balanced}" --interval 60
