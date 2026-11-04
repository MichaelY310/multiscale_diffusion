#!/usr/bin/env bash
set -euo pipefail

# Small, GPU-memory-safe search focused on the native RDD cascade:
#   8-stage boundary A -> 16-stage block route
#   16-stage final sparse A -> 32-stage block route
# It deliberately uses one physical GPU and only two classes.  This is a
# quality/kernel screen, not the final paper timing run.

ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-train/bin/python}"
OUT="${OUT:-${ROOT}/eval_results/rdd_packed_cascade_search}"
LOG="${LOG:-${ROOT}/eval_results/rdd_packed_cascade_search.log}"
MIN_FREE_MIB="${MIN_FREE_MIB:-24000}"
POLL_SECONDS="${POLL_SECONDS:-30}"
CLASSES="${CLASSES:-207,323}"
SEED="${SEED:-20260825}"

SETTINGS="${SETTINGS:-packed_cascade_safe_nocache,packed_cascade_safe_k2,packed_cascade_equal875_k2,packed_cascade_balanced_nocache,packed_cascade_balanced_k2,packed_cascade_moderate_k2,packed_cascade_middle_sparse_k2,packed_stage32_75_k2_control}"

mkdir -p "${OUT}" "$(dirname "${LOG}")"

while true; do
    read -r total used < <(
        nvidia-smi --id="${GPU}" \
            --query-gpu=memory.total,memory.used \
            --format=csv,noheader,nounits | tr -d ','
    )
    free_mib=$((total - used))
    if (( free_mib >= MIN_FREE_MIB )); then
        break
    fi
    printf '[%s] waiting: GPU %s has %s MiB free; require %s MiB\n' \
        "$(date --iso-8601=seconds)" "${GPU}" "${free_mib}" "${MIN_FREE_MIB}" | tee -a "${LOG}"
    sleep "${POLL_SECONDS}"
done

cd "${ROOT}"
printf '[%s] packed cascade search starts on physical GPU %s; classes=%s seed=%s\n' \
    "$(date --iso-8601=seconds)" "${GPU}" "${CLASSES}" "${SEED}" | tee -a "${LOG}"

# Correctness first.  These tests compare legacy and packed kernels to the same
# dense masked-softmax reference for both actual RDD block granularities.
env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    2d/attention_analysis/test_rdd_block_sparse_triton.py --dtype float32 2>&1 | tee -a "${LOG}"
env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    2d/attention_analysis/test_rdd_block_sparse_triton.py --dtype bfloat16 2>&1 | tee -a "${LOG}"

# Microbenchmark is recorded, but because another training job may be using the
# GPU it is diagnostic only.  Final timing is repeated after the GPU is idle.
env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    2d/attention_analysis/benchmark_rdd_block_sparse_triton.py \
    --dtype float32 --batch 2 --heads 16 --dim 72 \
    --tokens 64 256 --densities 0.625 0.75 0.875 0.9375 2>&1 | tee -a "${LOG}"

# Per-process cap is 25% of one 98-GiB GPU.  Together with the observed
# ~61-GiB resident trainer this leaves a safety margin and prevents this sampler
# from consuming memory needed by the existing job.
env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    2d/attention_analysis/sample_rdd_hierarchical_kernel.py \
    --unified-ckpt "${ROOT}/checkpoints/RDD-Unified-XL/0400000.pt" \
    --vae-path "${ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema" \
    --output-dir "${OUT}/seed_${SEED}" \
    --classes "${CLASSES}" \
    --cfg-scale 1.5 \
    --sample-steps 80,80,90 \
    --settings "${SETTINGS}" \
    --seed "${SEED}" \
    --timing-repeats 1 \
    --device cuda:0 \
    --memory-fraction 0.25 2>&1 | tee -a "${LOG}"

"${PYTHON}" 2d/attention_analysis/summarize_rdd_packed_cascade_search.py \
    "${OUT}/seed_${SEED}/metrics.json" 2>&1 | tee -a "${LOG}"

printf '[%s] packed cascade screen complete: %s\n' \
    "$(date --iso-8601=seconds)" "${OUT}/seed_${SEED}/metrics.json" | tee -a "${LOG}"
