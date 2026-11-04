#!/usr/bin/env bash
set -euo pipefail

# Exact VSA-style 12.5% block-retention ablation for the 2-D RDD sampler.
# The job only starts after the selected GPU's pre-existing memory use falls
# below the threshold; it never sends a signal to another process.

ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
GPU="${GPU:-1}"
MAX_PREEXISTING_GPU_MEMORY_MIB="${MAX_PREEXISTING_GPU_MEMORY_MIB:-70000}"
POLL_SECONDS="${POLL_SECONDS:-30}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-train/bin/python}"
OUT="${OUT:-${ROOT}/eval_results/rdd_hierarchical_kernel_vsa125_cfg15}"
LOG="${LOG:-${ROOT}/eval_results/rdd_hierarchical_kernel_vsa125_cfg15.log}"

mkdir -p "${OUT}" "$(dirname "${LOG}")"

while true; do
    used_mib="$(nvidia-smi --id="${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
    if [[ "${used_mib}" =~ ^[0-9]+$ ]] && (( used_mib <= MAX_PREEXISTING_GPU_MEMORY_MIB )); then
        break
    fi
    printf '[%s] waiting for GPU %s: %s MiB used (start threshold: %s MiB)\n' \
        "$(date --iso-8601=seconds)" "${GPU}" "${used_mib}" "${MAX_PREEXISTING_GPU_MEMORY_MIB}" | tee -a "${LOG}"
    sleep "${POLL_SECONDS}"
done

printf '[%s] starting exact-12.5%% attention ablation on physical GPU %s\n' \
    "$(date --iso-8601=seconds)" "${GPU}" | tee -a "${LOG}"

cd "${ROOT}"
env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    2d/attention_analysis/sample_rdd_hierarchical_kernel.py \
    --unified-ckpt "${ROOT}/checkpoints/RDD-Unified-XL/0400000.pt" \
    --vae-path "${ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema" \
    --output-dir "${OUT}" \
    --classes 207,282,323,817 \
    --cfg-scale 1.5 \
    --sample-steps 80,80,90 \
    --settings triton_vsa125_cascade_refresh1,triton_vsa125_cascade_score3 \
    --timing-repeats 1 \
    --device cuda:0 \
    --memory-fraction 0.30 2>&1 | tee -a "${LOG}"
