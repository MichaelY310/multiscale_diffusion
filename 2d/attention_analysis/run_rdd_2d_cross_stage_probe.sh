#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.venv-rdd-train/bin/python}"
PHYSICAL_GPU="${PROBE_GPU:-0}"
MIN_FREE_MIB="${MIN_FREE_MIB:-26000}"
OUTPUT="${OUTPUT_DIR:-${REPO_DIR}/eval_results/rdd_2d_cross_stage}"

used="$(nvidia-smi --id="${PHYSICAL_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
total="$(nvidia-smi --id="${PHYSICAL_GPU}" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')"
free=$((total - used))
if (( free < MIN_FREE_MIB )); then
    echo "ERROR: GPU ${PHYSICAL_GPU} has only ${free} MiB free; need ${MIN_FREE_MIB}." >&2
    exit 3
fi

unified="${REPO_DIR}/checkpoints/RDD-Unified-XL/0400000.pt"
multistage="${REPO_DIR}/checkpoints/RDD-Multistage-300-300-400"
vae="${REPO_DIR}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema"
imagenet="${REPO_DIR}/imagenet/ILSVRC/Data/CLS-LOC/train"
test -s "${unified}"
test -s "${multistage}/checkpoints_8x8/0400000.pt"
test -s "${multistage}/checkpoints_16x16/0400000.pt"
test -s "${multistage}/checkpoints_32x32/0400000.pt"
test -d "${vae}"
test -d "${imagenet}"
mkdir -p "${OUTPUT}/logs"

echo "RDD 2-D cross-stage probe"
echo "physical GPU=${PHYSICAL_GPU}, currently free=${free} MiB"
echo "output=${OUTPUT}"
CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" nice -n 15 "${PYTHON_BIN}" -u \
    "${SCRIPT_DIR}/probe_rdd_2d_cross_stage.py" \
    --unified-ckpt "${unified}" \
    --multistage-dir "${multistage}" \
    --vae-path "${vae}" \
    --imagenet-root "${imagenet}" \
    --output-dir "${OUTPUT}" \
    --device cuda:0 \
    --batch-size "${BATCH_SIZE:-4}" \
    --seeds "${SEEDS:-1234,5678}" \
    --memory-fraction "${MEMORY_FRACTION:-0.24}" \
    2>&1 | tee "${OUTPUT}/logs/probe.log"

