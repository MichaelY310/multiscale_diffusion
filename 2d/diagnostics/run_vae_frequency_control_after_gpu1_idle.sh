#!/usr/bin/env bash
# Use physical GPU 1 only after it is idle; never signals another process.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv-rdd-eval/bin/python}"
OUTPUT="${REPO_ROOT}/eval_results/diagnostics/vae_frequency_control"
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT}" logs

while nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
  | grep -Eq '^[[:space:]]*[0-9]+'; do
  echo "[$(date -Is)] physical GPU 1 still has a compute process; waiting"
  sleep 60
done

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
"${PYTHON}" -u 2d/diagnostics/reconstruct_npz_with_vae.py \
  --input-npz eval_results/reference/VIRTUAL_imagenet256_labeled.npz \
  --vae-path checkpoints/pretrained/stabilityai/sd-vae-ft-ema \
  --output-npz "${OUTPUT}/ImageNet_SD_VAE_mean_reconstruction_10000.npz" \
  --batch-size 32 \
  --max-images 10000 \
  --device cuda \
  2>&1 | tee logs/vae_frequency_control_reconstruction.log

# Avoid competing with the main ADM/FFT CPU job.  Reconstruction can finish
# first, but its frequency analysis waits for that CPU session to exit.
while tmux has-session -t rdd_stage_cpu_diag 2>/dev/null; do
  echo "[$(date -Is)] VAE reconstruction done; waiting for rdd_stage_cpu_diag before CPU FFT"
  sleep 60
done

export CUDA_VISIBLE_DEVICES=""
"${PYTHON}" -u 2d/diagnostics/analyze_image_frequency.py \
  --dataset "ImageNet_original=${REPO_ROOT}/eval_results/reference/VIRTUAL_imagenet256_labeled.npz" \
  --dataset "ImageNet_SD_VAE_mean_reconstruction=${OUTPUT}/ImageNet_SD_VAE_mean_reconstruction_10000.npz" \
  --reference-label ImageNet_original \
  --output-dir "${OUTPUT}/analysis" \
  --max-images 10000 \
  --batch-size 32 \
  --seed 20260814 \
  --fft-bins 64 \
  2>&1 | tee logs/vae_frequency_control_analysis.log
