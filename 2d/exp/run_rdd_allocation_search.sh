#!/usr/bin/env bash
# Run one sample_fid_unified JSONL on independent GPU ranks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv-rdd-eval/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: no Python interpreter; set PYTHON_BIN" >&2
  exit 2
fi

CONFIG="${CONFIG:?Set CONFIG to the generated JSONL file}"
if [[ ! -s "${CONFIG}" ]]; then echo "ERROR: missing config ${CONFIG}" >&2; exit 2; fi
GPUS="${GPUS:-0,1,2,3}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-5000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-16}"
SEED="${SEED:-0}"
VAE_PATH="${VAE_PATH:-${ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema}"
LOG_DIR="${LOG_DIR:-${ROOT}/eval_results/logs/allocation_search}"
mkdir -p "${LOG_DIR}"
if [[ ! -f "${VAE_PATH}/config.json" ]]; then
  echo "ERROR: local SD VAE not found at ${VAE_PATH}" >&2
  exit 2
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
WORLD_SIZE="${#GPU_ARRAY[@]}"
if (( WORLD_SIZE < 1 )); then echo "ERROR: GPUS is empty" >&2; exit 2; fi
RUN_TAG="$(basename "${CONFIG}" .jsonl)"
PIDS=()
for rank in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$rank]}"
  log="${LOG_DIR}/${RUN_TAG}_rank${rank}_gpu${gpu}.log"
  echo "rank ${rank}/${WORLD_SIZE} -> physical GPU ${gpu}; log ${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
    "${ROOT}/2d/sample_fid_unified.py" \
    --config "${CONFIG}" --rank "${rank}" --world-size "${WORLD_SIZE}" \
    --total-samples "${TOTAL_SAMPLES}" --batch-size "${BATCH_SIZE}" \
    --vae-batch-size "${VAE_BATCH_SIZE}" --seed "${SEED}" --vae-path "${VAE_PATH}" \
    >"${log}" 2>&1 &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
if (( status != 0 )); then
  echo "ERROR: at least one sampling rank failed; inspect ${LOG_DIR}/${RUN_TAG}_rank*.log" >&2
  exit "${status}"
fi
echo "All sampling ranks completed. Logs: ${LOG_DIR}/${RUN_TAG}_rank*.log"
