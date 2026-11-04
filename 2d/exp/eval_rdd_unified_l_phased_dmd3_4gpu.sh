#!/usr/bin/env bash
# Evaluate one RDD-Phased-DMD EMA export with exactly 3 conditional NFEs.
# Usage: bash 2d/exp/eval_rdd_unified_l_phased_dmd3_4gpu.sh CHECKPOINT [TAG]

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
CHECKPOINT="${1:?usage: $0 CHECKPOINT [TAG]}"
TAG="${2:-$(basename "${CHECKPOINT}" .pt)}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-32}"
VAE_PATH="${VAE_PATH:-${REPO_ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema}"
REF_NPZ="${RDD_REF_NPZ:-${REPO_ROOT}/eval_results/reference/VIRTUAL_imagenet256_labeled.npz}"
EVAL_ROOT="${EVAL_ROOT:-${REPO_ROOT}/eval_results/rdd_unified_l_phased_dmd3}"
SAMPLE_DIR="${EVAL_ROOT}/samples/${TAG}"
LOG_DIR="${EVAL_ROOT}/logs/${TAG}"
RESULT_FILE="${EVAL_ROOT}/metrics/${TAG}.txt"
NPZ_FILE="${EVAL_ROOT}/npz_tmp/${TAG}.npz"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
WORLD_SIZE="${#GPUS[@]}"
(( WORLD_SIZE > 0 )) || { echo "ERROR: GPU_LIST is empty" >&2; exit 1; }

cd "${REPO_ROOT}"
if [[ -f "${REPO_ROOT}/.venv-rdd-eval/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv-rdd-eval/bin/activate"
elif [[ -f "${REPO_ROOT}/2d/2d/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/2d/2d/bin/activate"
else
    echo "ERROR: evaluation environment not found" >&2
    exit 1
fi

SAMPLE_SCRIPT="${REPO_ROOT}/2d/sample_rdd_phased_dmd3.py"
PNG_TO_NPZ="${REPO_ROOT}/2d/png_to_npz.py"
EVALUATOR="${REPO_ROOT}/guided-diffusion/evaluations/evaluator.py"
for required in "${CHECKPOINT}" "${SAMPLE_SCRIPT}" "${PNG_TO_NPZ}" "${EVALUATOR}" "${REF_NPZ}"; do
    [[ -s "${required}" ]] || { echo "ERROR: missing ${required}" >&2; exit 1; }
done
[[ -d "${VAE_PATH}" ]] || { echo "ERROR: missing VAE ${VAE_PATH}" >&2; exit 1; }

if [[ -s "${REPO_ROOT}/eval_results/reference/classify_image_graph_def.pb" ]]; then
    INCEPTION_DIR="${REPO_ROOT}/eval_results/reference"
elif [[ -s "${REPO_ROOT}/classify_image_graph_def.pb" ]]; then
    INCEPTION_DIR="${REPO_ROOT}"
else
    echo "ERROR: classify_image_graph_def.pb is missing" >&2
    exit 1
fi

mkdir -p "${SAMPLE_DIR}" "${LOG_DIR}" "$(dirname "${RESULT_FILE}")" "$(dirname "${NPZ_FILE}")"

echo "RDD-Phased-DMD evaluation: ${TAG}"
echo "Checkpoint: ${CHECKPOINT}"
echo "GPUs: ${GPU_LIST}; samples: ${TOTAL_SAMPLES}; NFE: 3 (8+16+32)"

pids=()
for rank in "${!GPUS[@]}"; do
    gpu="${GPUS[$rank]}"
    CUDA_VISIBLE_DEVICES="${gpu}" python -u "${SAMPLE_SCRIPT}" \
        --checkpoint "${CHECKPOINT}" \
        --output-dir "${SAMPLE_DIR}" \
        --vae-path "${VAE_PATH}" \
        --model DiT-L/2 \
        --resolutions 32,16,8 \
        --stage-lengths 300,300,400 \
        --total-samples "${TOTAL_SAMPLES}" \
        --batch-size "${BATCH_SIZE}" \
        --vae-batch-size "${VAE_BATCH_SIZE}" \
        --rank "${rank}" --world-size "${WORLD_SIZE}" --seed 0 \
        > "${LOG_DIR}/gpu${gpu}.log" 2>&1 &
    pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
        echo "ERROR: sampler rank ${index} failed" >&2
        failed=1
    fi
done
(( failed == 0 )) || exit 1

count="$(find "${SAMPLE_DIR}" -maxdepth 1 -type f -name '*.png' | wc -l)"
[[ "${count}" -eq "${TOTAL_SAMPLES}" ]] || {
    echo "ERROR: expected ${TOTAL_SAMPLES} PNGs, found ${count}" >&2
    exit 1
}

python "${PNG_TO_NPZ}" "${SAMPLE_DIR}" --num-samples "${TOTAL_SAMPLES}" --output "${NPZ_FILE}"
set +e
(
    cd "${INCEPTION_DIR}"
    CUDA_VISIBLE_DEVICES="" python -u "${EVALUATOR}" "${REF_NPZ}" "${NPZ_FILE}"
) 2>&1 | tee "${RESULT_FILE}.tmp"
status="${PIPESTATUS[0]}"
set -e
rm -f -- "${NPZ_FILE}"
(( status == 0 )) || { mv -f -- "${RESULT_FILE}.tmp" "${RESULT_FILE}.failed.txt"; exit "${status}"; }
for metric in "Inception Score" FID sFID Precision Recall; do
    grep -qE "^${metric}:" "${RESULT_FILE}.tmp" || {
        mv -f -- "${RESULT_FILE}.tmp" "${RESULT_FILE}.failed.txt"
        echo "ERROR: missing ${metric}" >&2
        exit 1
    }
done
mv -f -- "${RESULT_FILE}.tmp" "${RESULT_FILE}"
echo "Complete metrics: ${RESULT_FILE}"

