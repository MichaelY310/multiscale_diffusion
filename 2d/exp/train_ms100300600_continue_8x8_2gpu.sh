#!/usr/bin/env bash
# Continue only the 8x8 DiT-B/2 stage of RDD-Multistage-100-300-600.
#
# Source checkpoint: 400K optimizer steps.
# Target checkpoint: 600K optimizer steps (200K additional updates).
# GPUs 2,3; per-rank micro-batch 32; 4-way accumulation => global batch 256.
# Data: native ILSVRC ImageFolder tree, with the same ADM center_crop_arr
# transform as the correct experiments. This is NOT the legacy pre-resized HF
# dataset that invalidated the old experiments.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_ACTIVATE="${VENV_ACTIVATE:-${ROOT_DIR}/.venv-rdd-train/bin/activate}"

if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "ERROR: missing environment: ${VENV_ACTIVATE}" >&2
  exit 1
fi
source "${VENV_ACTIVATE}"
cd "${ROOT_DIR}"

export TMPDIR="${TMPDIR:-${ROOT_DIR}/.tmp}"
mkdir -p "${TMPDIR}"

DATASET_DIR="${IMAGENET_TRAIN_DIR:-${ROOT_DIR}/imagenet/ILSVRC/Data/CLS-LOC/train}"
if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "ERROR: native ImageNet train directory not found: ${DATASET_DIR}" >&2
  exit 1
fi
class_count="$(find "${DATASET_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l)"
if [[ "${class_count}" != "1000" ]]; then
  echo "ERROR: expected 1000 ImageFolder class directories, found ${class_count}" >&2
  exit 1
fi

CKPT="${ROOT_DIR}/checkpoints/RDD-Multistage-100-300-600/checkpoints_8x8/0400000.pt"
VAE_DIR="${ROOT_DIR}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema"
RESULTS_DIR="${ROOT_DIR}/checkpoints/RDD-Multistage-100-300-600-8x8-continued-from400k"

if [[ ! -s "${CKPT}" ]]; then
  echo "ERROR: missing 8x8 checkpoint: ${CKPT}" >&2
  exit 1
fi
if [[ ! -d "${VAE_DIR}" ]]; then
  echo "ERROR: missing local SD VAE: ${VAE_DIR}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

echo "============================================================"
echo "Continue RDD-Multistage-100-300-600 8x8 stage: 400K -> 600K"
echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Batch: 32/GPU x 2 GPUs x accumulation 4 = effective global 256"
echo "Dataset: native ILSVRC ImageFolder + ADM crop (${class_count} classes)"
echo "Checkpoint: ${CKPT}"
echo "Results: ${RESULTS_DIR}/imagenet256-multiscale/checkpoints_8x8"
echo "============================================================"

torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  --master_port="${MASTER_PORT:-29612}" \
  "${ROOT_DIR}/2d/train_RDD_imagenet256.py" \
  --model-8 DiT-B/2 \
  --train-stages 8 \
  --results-dir "${RESULTS_DIR}" \
  --global-batch-size 256 \
  --micro-batch-size 32 \
  --gradient-accumulation-steps 4 \
  --num-workers 8 \
  --diffusion-steps 1000 \
  --stage-lengths 100,300,600 \
  --epochs 200 \
  --max-train-steps 600000 \
  --log-every 100 \
  --ckpt-every 25000 \
  --vae ema \
  --vae-path "${VAE_DIR}" \
  --global-seed 0 \
  --data-source imagenet \
  --data-path "${DATASET_DIR}" \
  --resume-8 "${CKPT}"
