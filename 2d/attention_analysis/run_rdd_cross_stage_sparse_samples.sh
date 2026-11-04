#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.venv-rdd-train/bin/python}"
PHYSICAL_GPU="${SAMPLE_GPU:-0}"
MIN_FREE_MIB="${MIN_FREE_MIB:-26000}"
OUTPUT="${OUTPUT_DIR:-${REPO_DIR}/eval_results/rdd_cross_stage_sparse_samples}"

used="$(nvidia-smi --id="${PHYSICAL_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
total="$(nvidia-smi --id="${PHYSICAL_GPU}" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')"
free=$((total - used))
if (( free < MIN_FREE_MIB )); then
    echo "ERROR: GPU ${PHYSICAL_GPU} has only ${free} MiB free; need ${MIN_FREE_MIB}." >&2
    exit 3
fi

mkdir -p "${OUTPUT}/logs"
echo "RDD cross-stage sparse sampling correctness oracle"
echo "GPU=${PHYSICAL_GPU}; free=${free} MiB; output=${OUTPUT}"
CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" nice -n 15 "${PYTHON_BIN}" -u \
    "${SCRIPT_DIR}/sample_rdd_cross_stage_sparse.py" \
    --unified-ckpt "${REPO_DIR}/checkpoints/RDD-Unified-XL/0400000.pt" \
    --multistage-dir "${REPO_DIR}/checkpoints/RDD-Multistage-300-300-400" \
    --vae-path "${REPO_DIR}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema" \
    --output-dir "${OUTPUT}" \
    --families "${FAMILIES:-unified,multistage}" \
    --classes "${CLASSES:-207,282,323,817}" \
    --cfg-scale "${CFG_SCALE:-1.5}" \
    --seed "${SEED:-20260819}" \
    --sample-steps "${SAMPLE_STEPS:-80,80,90}" \
    --settings "${SETTINGS:-all}" \
    --device cuda:0 \
    --memory-fraction "${MEMORY_FRACTION:-0.25}" \
    2>&1 | tee "${OUTPUT}/logs/sample.log"
