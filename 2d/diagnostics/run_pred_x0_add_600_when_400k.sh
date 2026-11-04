#!/usr/bin/env bash
# Run manually after RDD-Multistage-600-300-100 reaches the same 400K budget.
# The script refuses to use an early checkpoint and refuses to start while GPU0
# has any compute process.  It never kills another process.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv-rdd-eval/bin/python}"
CKPT_ROOT="${REPO_ROOT}/checkpoints/RDD-Multistage-600-300-100/imagenet256-multiscale"
cd "${REPO_ROOT}"

for resolution in 8 16 32; do
  checkpoint="${CKPT_ROOT}/checkpoints_${resolution}x${resolution}/0400000.pt"
  if [[ ! -s "${checkpoint}" ]]; then
    echo "ERROR: 600/300/100 is not yet at the comparable 400K checkpoint: ${checkpoint}" >&2
    exit 2
  fi
done
if nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
  | grep -Eq '^[[:space:]]*[0-9]+'; then
  echo "ERROR: physical GPU0 is busy; refusing to overlap with another process." >&2
  exit 3
fi

mkdir -p logs eval_results/diagnostics/stage_pred_x0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
"${PYTHON}" -u 2d/diagnostics/visualize_multistage_pred_x0.py \
  --config 2d/diagnostics/multistage_pred_x0_300_vs_100_vs_600.json \
  --output-dir eval_results/diagnostics/stage_pred_x0/300_vs_100_vs_600 \
  --cfg-scales 1.0,1.5 \
  --num-steps 80,80,90 \
  --seed 20260814 \
  --device cuda \
  2>&1 | tee logs/pred_x0_300_vs_100_vs_600.log

"${PYTHON}" -u 2d/diagnostics/build_stage_allocation_report.py \
  --repo-root "${REPO_ROOT}" \
  --bundle-dir "${REPO_ROOT}/eval_results/diagnostics/RDD_STAGE_ALLOCATION_REPORT_BUNDLE"
