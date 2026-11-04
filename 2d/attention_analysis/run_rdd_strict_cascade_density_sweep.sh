#!/usr/bin/env bash
set -euo pipefail

ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-train/bin/python}"
OUT="${OUT:-${ROOT}/eval_results/rdd_strict_cascade_density_sweep}"
LOG="${LOG:-${ROOT}/eval_results/rdd_strict_cascade_density_sweep.log}"
MIN_FREE_MIB="${MIN_FREE_MIB:-24000}"
POLL_SECONDS="${POLL_SECONDS:-30}"
SEED="${SEED:-20260828}"
SETTINGS="legacy_cascade_keep125_nocache,packed_cascade_keep125_nocache,legacy_cascade_keep1875_nocache,packed_cascade_keep1875_nocache,legacy_cascade_keep50_nocache,packed_cascade_keep50_nocache"

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
printf '[%s] strict cascade 12.5/18.75/50%% sweep starts\n' \
    "$(date --iso-8601=seconds)" | tee -a "${LOG}"
env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    2d/attention_analysis/sample_rdd_hierarchical_kernel.py \
    --unified-ckpt "${ROOT}/checkpoints/RDD-Unified-XL/0400000.pt" \
    --vae-path "${ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema" \
    --output-dir "${OUT}/seed_${SEED}" \
    --classes 207,282,323,817 \
    --cfg-scale 1.5 --sample-steps 80,80,90 \
    --settings "${SETTINGS}" --seed "${SEED}" --timing-repeats 5 \
    --device cuda:0 --memory-fraction 0.25 2>&1 | tee -a "${LOG}"

"${PYTHON}" 2d/attention_analysis/summarize_rdd_packed_cascade_search.py \
    "${OUT}/seed_${SEED}/metrics.json" 2>&1 | tee -a "${LOG}"
printf '[%s] strict cascade density sweep complete\n' \
    "$(date --iso-8601=seconds)" | tee -a "${LOG}"
