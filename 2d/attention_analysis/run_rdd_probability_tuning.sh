#!/usr/bin/env bash
set -euo pipefail

ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
GPU="${GPU:-1}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-train/bin/python}"
MAX_PREEXISTING_GPU_MEMORY_MIB="${MAX_PREEXISTING_GPU_MEMORY_MIB:-1000}"
POLL_SECONDS="${POLL_SECONDS:-30}"
BASE_OUT="${BASE_OUT:-${ROOT}/eval_results/rdd_probability_tuning16}"
LOG="${LOG:-${ROOT}/eval_results/rdd_probability_tuning16.log}"
SEEDS="${SEEDS:-20260822 20260823 20260824}"
CLASSES="${CLASSES:-1,25,130,207,282,323,388,417,497,546,574,701,817,850,932,985}"
SETTINGS="fixed_16_32_k2,fixed_16_32_k3,fixed_16k2_32k3,fixed_16k3_32k2,fixed_16_32_k3_first7,fixed_16_32_k3_first14,fixed_16_32_k3_first21,fixed_16_32_k4_first7"

mkdir -p "${BASE_OUT}" "$(dirname "${LOG}")"
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
    printf '[%s] probability tuning seed=%s GPU=%s\n' \
        "$(date --iso-8601=seconds)" "${seed}" "${GPU}" | tee -a "${LOG}"
    env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
        2d/attention_analysis/sample_rdd_adaptive_probability_cache.py \
        --unified-ckpt "${ROOT}/checkpoints/RDD-Unified-XL/0400000.pt" \
        --vae-path "${ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema" \
        --output-dir "${BASE_OUT}/seed_${seed}" \
        --classes "${CLASSES}" \
        --cfg-scale 1.5 \
        --sample-steps 80,80,90 \
        --settings "${SETTINGS}" \
        --seed "${seed}" \
        --timing-repeats 3 \
        --device cuda:0 \
        --memory-fraction 0.45 2>&1 | tee -a "${LOG}"
done

printf '[%s] probability tuning complete\n' "$(date --iso-8601=seconds)" | tee -a "${LOG}"
