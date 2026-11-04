#!/usr/bin/env bash
# Wait until physical GPU 0 has no compute process, then run the small fixed-seed
# 300/300/400 versus 100/300/600 pred-x0 diagnostic.  It never kills or signals
# any process and therefore cannot stop another user's job.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv-rdd-eval/bin/python}"
cd "${REPO_ROOT}"
mkdir -p logs eval_results/diagnostics/stage_pred_x0

while nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
  | grep -Eq '^[[:space:]]*[0-9]+'; do
  echo "[$(date -Is)] physical GPU 0 still has a compute process; waiting"
  sleep 60
done

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
exec "${PYTHON}" -u 2d/diagnostics/visualize_multistage_pred_x0.py \
  --config 2d/diagnostics/multistage_pred_x0_300_vs_100.json \
  --output-dir eval_results/diagnostics/stage_pred_x0/300_vs_100 \
  --cfg-scales 1.0,1.5 \
  --num-steps 80,80,90 \
  --seed 20260814 \
  --device cuda
