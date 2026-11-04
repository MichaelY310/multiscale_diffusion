#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv-rdd-eval/bin/python}"
cd "${REPO_ROOT}"
mkdir -p logs eval_results/diagnostics/class_consistency

while tmux has-session -t local_solver_after_teacher 2>/dev/null; do
  echo "[$(date -Is)] waiting for local_solver_after_teacher"
  sleep 60
done
while tmux has-session -t rdd_stage_cpu_diag 2>/dev/null; do
  echo "[$(date -Is)] waiting for rdd_stage_cpu_diag to pack continued-model NPZs"
  sleep 60
done
while nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
  | grep -Eq '^[[:space:]]*[0-9]+'; do
  echo "[$(date -Is)] physical GPU 0 still has a compute process; waiting"
  sleep 60
done

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
exec "${PYTHON}" -u 2d/diagnostics/evaluate_generated_class_consistency.py \
  --dataset "RDD_100_300_600_cfg1=${REPO_ROOT}/eval_results/npz/ms100300600_400k_808090_ddpm_cfg1_noisefilling_samples.npz" \
  --dataset "RDD_100_300_600_cfg1p5=${REPO_ROOT}/eval_results/npz/ms100300600_400k_808090_ddpm_cfg15_noisefilling_samples.npz" \
  --dataset "RDD_100_300_600_8x8continued600K_cfg1=${REPO_ROOT}/eval_results/npz/ms100300600_8x8continued600k_16x32orig400k_808090_ddpm_cfg1_noisefilling_samples.npz" \
  --dataset "RDD_100_300_600_8x8continued600K_cfg1p5=${REPO_ROOT}/eval_results/npz/ms100300600_8x8continued600k_16x32orig400k_808090_ddpm_cfg15_noisefilling_samples.npz" \
  --output-dir eval_results/diagnostics/class_consistency/100_300_600 \
  --total-samples 50000 \
  --dataset-world-sizes 2,2,4,4 \
  --dataset-sample-batch-sizes 512,512,512,512 \
  --classifier-batch-size 256 \
  --seed 0 \
  --num-classes 1000 \
  --device cuda
