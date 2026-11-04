#!/usr/bin/env bash
set -euo pipefail

ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-train/bin/python}"
OUT="${OUT:-${ROOT}/eval_results/rdd_hybrid32_confirm}"
LOG="${LOG:-${ROOT}/eval_results/rdd_hybrid32_confirm.log}"
MIN_FREE_MIB="${MIN_FREE_MIB:-24000}"
POLL_SECONDS="${POLL_SECONDS:-30}"
SEED="${SEED:-20260827}"
SETTINGS="${SETTINGS:-triton_dense_score2_all,legacy_cascade_safe_k2,packed_cascade_safe_k2,hybrid32_cascade_safe_k2,hybrid32_cascade_balanced_k2}"

mkdir -p "${OUT}" "$(dirname "${LOG}")"
while true; do
    read -r total used < <(
        nvidia-smi --id="${GPU}" --query-gpu=memory.total,memory.used \
            --format=csv,noheader,nounits | tr -d ','
    )
    free_mib=$((total - used))
    if (( free_mib >= MIN_FREE_MIB )); then break; fi
    printf '[%s] waiting: GPU %s has %s MiB free; require %s MiB\n' \
        "$(date --iso-8601=seconds)" "${GPU}" "${free_mib}" "${MIN_FREE_MIB}" | tee -a "${LOG}"
    sleep "${POLL_SECONDS}"
done

cd "${ROOT}"
printf '[%s] hybrid 16-legacy / 32-packed32 confirmation starts\n' \
    "$(date --iso-8601=seconds)" | tee -a "${LOG}"
env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    2d/attention_analysis/test_rdd_block_sparse_triton.py --dtype float32 2>&1 | tee -a "${LOG}"
env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    2d/attention_analysis/test_rdd_block_sparse_triton.py --dtype bfloat16 2>&1 | tee -a "${LOG}"

env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    2d/attention_analysis/sample_rdd_hierarchical_kernel.py \
    --unified-ckpt "${ROOT}/checkpoints/RDD-Unified-XL/0400000.pt" \
    --vae-path "${ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema" \
    --output-dir "${OUT}/seed_${SEED}" \
    --classes 207,282,323,817 \
    --cfg-scale 1.5 --sample-steps 80,80,90 \
    --settings "${SETTINGS}" --seed "${SEED}" \
    --timing-repeats 5 --device cuda:0 --memory-fraction 0.25 2>&1 | tee -a "${LOG}"

"${PYTHON}" 2d/attention_analysis/summarize_rdd_packed_cascade_search.py \
    "${OUT}/seed_${SEED}/metrics.json" 2>&1 | tee -a "${LOG}"
printf '[%s] hybrid32 confirmation complete\n' "$(date --iso-8601=seconds)" | tee -a "${LOG}"
