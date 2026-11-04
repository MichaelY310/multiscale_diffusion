#!/usr/bin/env bash
# Serialize the teacher-forced diagnostic after the pred-x0 trajectory job so
# two watchers cannot claim GPU 0 at the same instant.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv-rdd-eval/bin/python}"
cd "${REPO_ROOT}"
mkdir -p logs eval_results/diagnostics/teacher_forced_pred_x0

while tmux has-session -t predx0_300_vs_100_after_gpu0_idle 2>/dev/null; do
  echo "[$(date -Is)] waiting for predx0_300_vs_100_after_gpu0_idle"
  sleep 60
done
while nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
  | grep -Eq '^[[:space:]]*[0-9]+'; do
  echo "[$(date -Is)] physical GPU 0 still has a compute process; waiting"
  sleep 60
done

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
exec "${PYTHON}" -u 2d/diagnostics/profile_teacher_forced_pred_x0.py \
  --config 2d/diagnostics/multistage_pred_x0_300_vs_100.json \
  --imagenet-root imagenet \
  --vae-path checkpoints/pretrained/stabilityai/sd-vae-ft-ema \
  --output-dir eval_results/diagnostics/teacher_forced_pred_x0/300_vs_100 \
  --num-images 256 \
  --batch-size 16 \
  --num-workers 8 \
  --seed 20260814 \
  --device cuda
