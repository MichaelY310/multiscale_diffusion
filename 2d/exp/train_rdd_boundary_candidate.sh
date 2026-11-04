#!/usr/bin/env bash
# Train one native-ImageNet RDD boundary candidate. Supports 32,16,8 and 32,8.
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

STAGE_RESOLUTIONS="${STAGE_RESOLUTIONS:-32,16,8}"
STAGE_LENGTHS="${STAGE_LENGTHS:-300,300,400}"
GPUS="${GPUS:-0,1,2,3}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-50000}"
CKPT_EVERY="${CKPT_EVERY:-50000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MASTER_PORT="${MASTER_PORT:-29641}"
RESULTS_NAME="${RESULTS_NAME:-RDD-boundary-${STAGE_RESOLUTIONS//,/-}-L${STAGE_LENGTHS//,/-}}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT}/checkpoints/rdd_boundary_search/${RESULTS_NAME}}"
VAE_PATH="${VAE_PATH:-${ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema}"

if [[ -z "${DATA_PATH:-}" ]]; then
  for candidate in \
    "${ROOT}/imagenet/ILSVRC/Data/CLS-LOC/train" \
    "${ROOT}/imagenet/train" \
    "${ROOT}/imagenet"; do
    if [[ -d "${candidate}" ]]; then DATA_PATH="${candidate}"; break; fi
  done
fi
if [[ -z "${DATA_PATH:-}" || ! -d "${DATA_PATH}" ]]; then
  echo "ERROR: native ImageNet train directory not found; set DATA_PATH" >&2
  exit 2
fi
CLASS_COUNT="$(find "${DATA_PATH}" -mindepth 1 -maxdepth 1 -type d | wc -l)"
if [[ "${CLASS_COUNT}" != "1000" ]]; then
  echo "ERROR: expected 1000 ImageFolder class directories under ${DATA_PATH}, found ${CLASS_COUNT}" >&2
  exit 2
fi
if [[ ! -f "${VAE_PATH}/config.json" ]]; then
  echo "ERROR: local SD VAE not found at ${VAE_PATH}" >&2
  exit 2
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
NPROC="${#GPU_ARRAY[@]}"
if (( NPROC < 1 )); then echo "ERROR: GPUS is empty" >&2; exit 2; fi
MICRO_GLOBAL=$(( MICRO_BATCH_SIZE * NPROC ))
if (( GLOBAL_BATCH_SIZE % MICRO_GLOBAL != 0 )); then
  echo "ERROR: GLOBAL_BATCH_SIZE must be divisible by MICRO_BATCH_SIZE * GPU count" >&2
  exit 2
fi
GRAD_ACCUM=$(( GLOBAL_BATCH_SIZE / MICRO_GLOBAL ))

IFS=',' read -r -a RES_ARRAY <<< "${STAGE_RESOLUTIONS}"
IFS=',' read -r -a LEN_ARRAY <<< "${STAGE_LENGTHS}"
if (( ${#RES_ARRAY[@]} != ${#LEN_ARRAY[@]} )); then
  echo "ERROR: STAGE_RESOLUTIONS and STAGE_LENGTHS must have equal lengths" >&2
  exit 2
fi
TOTAL=0
for value in "${LEN_ARRAY[@]}"; do TOTAL=$(( TOTAL + value )); done
if (( TOTAL != 1000 )); then echo "ERROR: STAGE_LENGTHS must sum to 1000" >&2; exit 2; fi

RESUME_ARGS=()
if [[ -n "${RESUME_STEP:-}" ]]; then
  RESUME_BASE="${RESUME_DIR:-${RESULTS_DIR}/imagenet256-multiscale}"
  if [[ -d "${RESUME_BASE}/imagenet256-multiscale" ]]; then
    RESUME_BASE="${RESUME_BASE}/imagenet256-multiscale"
  fi
  for resolution in "${RES_ARRAY[@]}"; do
    checkpoint="${RESUME_BASE}/checkpoints_${resolution}x${resolution}/${RESUME_STEP}.pt"
    if [[ ! -s "${checkpoint}" ]]; then
      echo "ERROR: missing resume checkpoint ${checkpoint}" >&2
      exit 2
    fi
    RESUME_ARGS+=("--resume-${resolution}" "${checkpoint}")
  done
fi

mkdir -p "${RESULTS_DIR}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export TMPDIR="${TMPDIR:-${ROOT}/.tmp}"
mkdir -p "${TMPDIR}"

echo "============================================================"
echo "RDD boundary candidate: ${RESULTS_NAME}"
echo "Topology/lengths: ${STAGE_RESOLUTIONS} / ${STAGE_LENGTHS} (finest->coarsest)"
echo "GPUs: ${GPUS}; micro ${MICRO_BATCH_SIZE} x ${NPROC} x accum ${GRAD_ACCUM} = ${GLOBAL_BATCH_SIZE}"
echo "Native ImageNet: ${DATA_PATH}"
echo "Max optimizer step: ${MAX_TRAIN_STEPS}; output: ${RESULTS_DIR}"
echo "============================================================"

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
  "${ROOT}/2d/train_RDD_imagenet256.py" \
  --stage-resolutions "${STAGE_RESOLUTIONS}" \
  --stage-lengths "${STAGE_LENGTHS}" \
  --train-stages all \
  --model-32 DiT-XL/2 --model-16 DiT-L/2 --model-8 DiT-B/2 \
  --results-dir "${RESULTS_DIR}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --micro-batch-size "${MICRO_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRAD_ACCUM}" \
  --max-train-steps "${MAX_TRAIN_STEPS}" \
  --ckpt-every "${CKPT_EVERY}" --log-every 100 --num-workers "${NUM_WORKERS}" \
  --diffusion-steps 1000 --epochs 2000 --global-seed 0 \
  --timestep-sampler uniform --loss-weighting none \
  --vae ema --vae-path "${VAE_PATH}" \
  --data-source imagenet --data-path "${DATA_PATH}" \
  "${RESUME_ARGS[@]}"
