#!/usr/bin/env bash
set -euo pipefail

ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
GPU="${GPU:-1}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-train/bin/python}"
MAX_PREEXISTING_GPU_MEMORY_MIB="${MAX_PREEXISTING_GPU_MEMORY_MIB:-1000}"
POLL_SECONDS="${POLL_SECONDS:-30}"
KERNEL_OUT="${KERNEL_OUT:-${ROOT}/eval_results/rdd_training_free_kernel_sweep}"
CFG_OUT="${CFG_OUT:-${ROOT}/eval_results/rdd_cfg_probability_share_sweep}"
LOG="${LOG:-${ROOT}/eval_results/rdd_training_free_kernel_cfg_sweep.log}"
SEEDS="${SEEDS:-20260819 20260820 20260821}"

mkdir -p "${KERNEL_OUT}" "${CFG_OUT}" "$(dirname "${LOG}")"

while true; do
    used_mib="$(nvidia-smi --id="${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
    if [[ "${used_mib}" =~ ^[0-9]+$ ]] && (( used_mib <= MAX_PREEXISTING_GPU_MEMORY_MIB )); then
        break
    fi
    printf '[%s] waiting for physical GPU %s: %s MiB used\n' \
        "$(date --iso-8601=seconds)" "${GPU}" "${used_mib}" | tee -a "${LOG}"
    sleep "${POLL_SECONDS}"
done

cd "${ROOT}"
for seed in ${SEEDS}; do
    printf '[%s] true-kernel training-free sweep seed=%s GPU=%s\n' \
        "$(date --iso-8601=seconds)" "${seed}" "${GPU}" | tee -a "${LOG}"
    env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
        2d/attention_analysis/sample_rdd_training_free_kernel_sweep.py \
        --unified-ckpt "${ROOT}/checkpoints/RDD-Unified-XL/0400000.pt" \
        --vae-path "${ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema" \
        --output-dir "${KERNEL_OUT}/seed_${seed}" \
        --classes 207,282,323,817 \
        --cfg-scale 1.5 \
        --sample-steps 80,80,90 \
        --seed "${seed}" \
        --timing-repeats 3 \
        --device cuda:0 \
        --memory-fraction 0.45 2>&1 | tee -a "${LOG}"

    printf '[%s] CFG probability-share sweep seed=%s GPU=%s\n' \
        "$(date --iso-8601=seconds)" "${seed}" "${GPU}" | tee -a "${LOG}"
    env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
        2d/attention_analysis/sample_rdd_cfg_probability_share.py \
        --unified-ckpt "${ROOT}/checkpoints/RDD-Unified-XL/0400000.pt" \
        --vae-path "${ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema" \
        --output-dir "${CFG_OUT}/seed_${seed}" \
        --classes 207,282,323,817 \
        --cfg-scale 1.5 \
        --sample-steps 80,80,90 \
        --seed "${seed}" \
        --timing-repeats 3 \
        --device cuda:0 \
        --memory-fraction 0.45 2>&1 | tee -a "${LOG}"
done

printf '[%s] true-kernel and CFG sweeps complete\n' "$(date --iso-8601=seconds)" | tee -a "${LOG}"
