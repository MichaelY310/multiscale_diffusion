#!/usr/bin/env bash
# Data-only initial boundary prediction for 32/16/8 RDD.
# Safety: this launcher only considers physical GPUs 0,1,2,3. It never kills
# or pauses any process and never touches GPUs 4,5,6,7.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv-rdd-train/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: no Python interpreter; set PYTHON_BIN" >&2
  exit 2
fi

if [[ -z "${DATA_PATH:-}" ]]; then
  for candidate in \
    "${ROOT}/imagenet/ILSVRC/Data/CLS-LOC/train" \
    "${ROOT}/imagenet/train" \
    "${ROOT}/imagenet"; do
    if [[ -d "${candidate}" ]]; then DATA_PATH="${candidate}"; break; fi
  done
fi
if [[ -z "${DATA_PATH:-}" || ! -d "${DATA_PATH}" ]]; then
  echo "ERROR: native ImageNet ImageFolder not found; set DATA_PATH" >&2
  exit 2
fi
CLASS_COUNT="$(find "${DATA_PATH}" -mindepth 1 -maxdepth 1 -type d | wc -l)"
if [[ "${CLASS_COUNT}" != "1000" ]]; then
  echo "ERROR: expected 1000 class directories in ${DATA_PATH}, found ${CLASS_COUNT}" >&2
  exit 2
fi

VAE_PATH="${VAE_PATH:-${ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema}"
if [[ ! -f "${VAE_PATH}/config.json" ]]; then
  echo "ERROR: local SD VAE not found at ${VAE_PATH}" >&2
  exit 2
fi

# Explicit GPU overrides are accepted only for the user-approved set 0..3.
if [[ -n "${GPU:-}" ]]; then
  case "${GPU}" in 0|1|2|3) ;; *) echo "ERROR: GPU must be one of 0,1,2,3" >&2; exit 2 ;; esac
  FREE_MIB="$(nvidia-smi --id="${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
else
  GPU=""
  FREE_MIB=-1
  for candidate in 0 1 2 3; do
    candidate_free="$(nvidia-smi --id="${candidate}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if (( candidate_free > FREE_MIB )); then
      GPU="${candidate}"
      FREE_MIB="${candidate_free}"
    fi
  done
fi

if [[ -z "${BATCH_SIZE:-}" ]]; then
  if (( FREE_MIB >= 16000 )); then BATCH_SIZE=32
  elif (( FREE_MIB >= 10000 )); then BATCH_SIZE=16
  elif (( FREE_MIB >= 6000 )); then BATCH_SIZE=8
  elif (( FREE_MIB >= 3500 )); then BATCH_SIZE=4
  else BATCH_SIZE=1
  fi
fi

NUM_IMAGES="${NUM_IMAGES:-10000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
OUTPUT="${OUTPUT:-${ROOT}/eval_results/rdd_boundary_prior_3stage.json}"
LOG="${LOG:-${ROOT}/eval_results/logs/rdd_boundary_prior_3stage.log}"
mkdir -p "$(dirname "${OUTPUT}")" "$(dirname "${LOG}")"

echo "============================================================"
echo "RDD initial 3-stage boundary prediction (data-only)"
echo "Physical GPU: ${GPU} (selected only from 0,1,2,3)"
echo "Free memory at launch: ${FREE_MIB} MiB; VAE batch: ${BATCH_SIZE}"
echo "Images: ${NUM_IMAGES}; ImageNet: ${DATA_PATH}"
echo "Output: ${OUTPUT}"
echo "Log: ${LOG}"
echo "No training, no process termination, GPUs 4-7 untouched."
echo "============================================================"

export CUDA_VISIBLE_DEVICES="${GPU}"
exec "${PYTHON_BIN}" -u "${ROOT}/2d/exp/estimate_rdd_boundaries.py" \
  --data-path "${DATA_PATH}" \
  --vae-path "${VAE_PATH}" \
  --num-images "${NUM_IMAGES}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --device cuda \
  --posterior mean \
  --snr-threshold 1.0 \
  --thresholds 0.5,1.0,2.0 \
  --bootstrap-replicates 200 \
  --bootstrap-seed 0 \
  --grid 50 \
  --output "${OUTPUT}" \
  2>&1 | tee "${LOG}"
