#!/usr/bin/env bash
# Train one paper-parity RDD-FM Unified DiT-L/2 allocation on four GPUs.
#
# DDPM_ALLOCATION is always written in resolution order (fine/mid/coarse):
#   300,300,400 -> FM data-time CLI 400,300,300 -> [0,.4,.7,1]
#   400,300,300 -> FM data-time CLI 300,300,400 -> [0,.3,.6,1]

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
DDPM_ALLOCATION="${DDPM_ALLOCATION:?set DDPM_ALLOCATION to 300,300,400 or 400,300,300}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?set exactly four physical GPU ids}"

case "${DDPM_ALLOCATION}" in
    300,300,400)
        RUN_SUFFIX="300-300-400"
        FM_STAGE_LENGTHS="400,300,300"
        EXPECTED_BOUNDARIES="[0.0, 0.4, 0.7, 1.0]"
        ;;
    400,300,300)
        RUN_SUFFIX="400-300-300"
        FM_STAGE_LENGTHS="300,300,400"
        EXPECTED_BOUNDARIES="[0.0, 0.3, 0.6, 1.0]"
        ;;
    *)
        echo "ERROR: unsupported DDPM_ALLOCATION=${DDPM_ALLOCATION}" >&2
        exit 1
        ;;
esac

readonly GLOBAL_BATCH_SIZE=256
readonly MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-256}"
readonly MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-400000}"
readonly NUM_WORKERS="${NUM_WORKERS:-4}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/checkpoints/RDD-FM-Unified-L-corrected-${RUN_SUFFIX}}"
CHECKPOINT_DIR="${RESULTS_DIR}/imagenet256-fm-unified/checkpoints"
TRAIN_LOG="${TRAIN_LOG:-${REPO_ROOT}/logs/train_rdd_fm_unified_l_corrected_${RUN_SUFFIX}.log}"
TRAIN_ENTRY="${REPO_ROOT}/2d/train_RDD_FM_imagenet256_unified.py"
CONTROL_DIR="${RESULTS_DIR}/.launcher-control"

mkdir -p "${RESULTS_DIR}" "${CONTROL_DIR}" "${REPO_ROOT}/logs"
command -v flock >/dev/null 2>&1 || { echo "ERROR: flock is required" >&2; exit 1; }
exec 9>"${CONTROL_DIR}/launcher.lock"
flock -n 9 || { echo "ERROR: another launcher owns ${RESULTS_DIR}" >&2; exit 1; }

if [[ -f "${REPO_ROOT}/.venv-rdd-train/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv-rdd-train/bin/activate"
elif [[ -f "${REPO_ROOT}/2d/2d/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/2d/2d/bin/activate"
else
    echo "ERROR: training environment not found" >&2
    exit 1
fi

for command_name in python3 torchrun tee; do
    command -v "${command_name}" >/dev/null 2>&1 || {
        echo "ERROR: missing command ${command_name}" >&2
        exit 1
    }
done
[[ -f "${TRAIN_ENTRY}" ]] || { echo "ERROR: missing ${TRAIN_ENTRY}" >&2; exit 1; }

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#GPU_IDS[@]} != 4 )); then
    echo "ERROR: exactly four GPUs are required; got ${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
fi
if (( GLOBAL_BATCH_SIZE % MICRO_BATCH_SIZE != 0 || MICRO_BATCH_SIZE % 4 != 0 )); then
    echo "ERROR: MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE} must divide 256 and be divisible by 4" >&2
    exit 1
fi

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/imagenet}"
if [[ -d "${DATA_ROOT}/ILSVRC/Data/CLS-LOC/train" ]]; then
    DATA_PATH="${DATA_ROOT}/ILSVRC/Data/CLS-LOC/train"
elif [[ -d "${DATA_ROOT}/train" ]]; then
    DATA_PATH="${DATA_ROOT}/train"
else
    echo "ERROR: original ImageNet train directory not found below ${DATA_ROOT}" >&2
    exit 1
fi

VAE_PATH="${VAE_PATH:-${REPO_ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema}"
[[ -d "${VAE_PATH}" ]] || { echo "ERROR: local SD VAE missing: ${VAE_PATH}" >&2; exit 1; }

if [[ -z "${MASTER_PORT:-}" ]]; then
    MASTER_PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
fi

RESUME_ARGS=()
if [[ -n "${RESUME_CKPT:-}" ]]; then
    [[ -s "${RESUME_CKPT}" ]] || { echo "ERROR: missing RESUME_CKPT=${RESUME_CKPT}" >&2; exit 1; }
    RESUME_ARGS=(--resume "${RESUME_CKPT}")
elif [[ "${AUTO_RESUME:-1}" == "1" && -d "${CHECKPOINT_DIR}" ]]; then
    LATEST_CKPT="$(find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name '[0-9]*.pt' -print | sort -V | tail -n 1)"
    [[ -z "${LATEST_CKPT}" ]] || RESUME_ARGS=(--resume "${LATEST_CKPT}")
fi

TRAIN_ARGS=(
    --model DiT-L/2
    --resolutions 32,16,8
    --stage-lengths "${FM_STAGE_LENGTHS}"
    --diffusion-steps 1000
    --results-dir "${RESULTS_DIR}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --max-train-steps "${MAX_TRAIN_STEPS}"
    --num-workers "${NUM_WORKERS}"
    --no-pin-memory
    --epochs 1400
    --log-every 100
    --ckpt-every 50000
    --vae ema
    --vae-path "${VAE_PATH}"
    --global-seed 0
    --time-sampler uniform
    --data-source imagenet
    --data-path "${DATA_PATH}"
    "${RESUME_ARGS[@]}"
)

echo "============================================================"
echo "RDD-FM Unified DiT-L/2"
echo "DDPM resolution allocation (fine/mid/coarse): ${DDPM_ALLOCATION}"
echo "FM data-time allocation (coarse/mid/fine): ${FM_STAGE_LENGTHS}"
echo "Expected FM boundaries: ${EXPECTED_BOUNDARIES}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Global batch: 256; per-GPU micro batch: $((MICRO_BATCH_SIZE / 4)); accumulation: $((256 / MICRO_BATCH_SIZE))"
echo "Original ImageNet: ${DATA_PATH}"
echo "Target optimizer step: ${MAX_TRAIN_STEPS}"
echo "Results: ${RESULTS_DIR}"
echo "Log: ${TRAIN_LOG}"
if (( ${#RESUME_ARGS[@]} > 0 )); then echo "Resume: ${RESUME_ARGS[1]}"; else echo "Resume: none"; fi
echo "============================================================"

cd "${REPO_ROOT}"
set -o pipefail
torchrun --nnodes=1 --nproc_per_node=4 --master_port="${MASTER_PORT}" \
    "${TRAIN_ENTRY}" "${TRAIN_ARGS[@]}" 2>&1 | tee -a "${TRAIN_LOG}"

