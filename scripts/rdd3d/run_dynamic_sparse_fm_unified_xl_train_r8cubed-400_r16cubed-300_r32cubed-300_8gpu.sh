#!/usr/bin/env bash
set -Eeuo pipefail

# RDD-3D Flow Matching paper-parity run.
#
# Transferable settings are intentionally matched to the corrected 2-D
# RDD-FM Unified XL/2 run:
#   * one unified model, trained from scratch;
#   * fine/middle/coarse allocation 300/300/400;
#   * FM data-time boundaries [0, 0.4, 0.7, 1.0];
#   * coarse -> middle -> fine stages all contribute on every shape;
#   * uniform global time inside every fixed stage interval;
#   * global batch 256, AdamW(lr=1e-4, wd=0), EMA 0.9999;
#   * 400K optimizer updates, seed 0.
#
# There is deliberately no stage embedding, endpoint oversampling, generated
# parent replay, empty-parent augmentation, positive-class reweighting, or
# band-width loss reweighting.  The only additional head is the occupancy head
# required by sparse 3-D pruning.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${RDD3D_FM_PYTHON:-${ROOT}/.venv_rdd3d/bin/python}"
TRAIN_ENTRY="${RDD3D_FM_TRAIN_ENTRY:-${ROOT}/RDD_DiT_3D/train_dynamic_sparse_rdd.py}"
TRAIN_MODULE_DIR="$(cd "$(dirname "${TRAIN_ENTRY}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NPROC="${#GPU_IDS[@]}"
if (( NPROC != 8 )); then
    echo "ERROR: the paper-parity launcher requires exactly 8 GPUs; got ${CUDA_VISIBLE_DEVICES}" >&2
    exit 2
fi

DATA_ROOT="${RDD3D_FM_DATA_ROOT:-${ROOT}/datasets/rdd3d_scale1k/ObjaverseXL_github_train_1280}"
SPLIT_ROOT="${RDD3D_FM_SPLIT_ROOT:-${DATA_ROOT}/rdd_dynamic_split_896train_96dev_96test_seed104729}"
TRAIN_LATENTS="${RDD3D_FM_TRAIN_LATENTS:-${SPLIT_ROOT}/train_latents}"
CONDITIONS="${RDD3D_FM_CONDITIONS:-${SPLIT_ROOT}/conditions_dinov2_vits14}"

SMOKE_TEST="${RDD3D_FM_SMOKE_TEST:-0}"
MODEL_WIDTH="${RDD3D_FM_WIDTH:-1152}"
MODEL_DEPTH="${RDD3D_FM_DEPTH:-28}"
GLOBAL_BATCH_SIZE="${RDD3D_FM_GLOBAL_BATCH_SIZE:-256}"
TOTAL_STEPS="${RDD3D_FM_TOTAL_STEPS:-400000}"
MAX_ASSETS="${RDD3D_FM_MAX_ASSETS:-896}"
SAVE_EVERY="${RDD3D_FM_SAVE_EVERY:-50000}"
LATEST_EVERY="${RDD3D_FM_LATEST_EVERY:-1000}"
LOG_EVERY="${RDD3D_FM_LOG_EVERY:-100}"
AMP_MODE="${RDD3D_FM_AMP:-none}"

RUN_NAME="dynamic_sparse_fm_unified_xl_300300400_8gpu"
if [[ "${SMOKE_TEST}" == "1" ]]; then
    # One actual optimizer update with the production model.  The smaller
    # effective batch tests forward/backward/DDP and peak per-shape memory;
    # accumulation does not retain graphs between microsteps.
    GLOBAL_BATCH_SIZE="${RDD3D_FM_SMOKE_GLOBAL_BATCH_SIZE:-8}"
    TOTAL_STEPS="${RDD3D_FM_SMOKE_STEPS:-1}"
    MAX_ASSETS="${RDD3D_FM_SMOKE_MAX_ASSETS:-32}"
    # A memory/forward/backward smoke test must not serialize the ~8.4 GB
    # model+EMA+Adam checkpoint.
    SAVE_EVERY=0
    LATEST_EVERY=0
    LOG_EVERY=1
    SMOKE_SAVE_ARGS=(--no-save-final)
    RUN_NAME="${RUN_NAME}_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
else
    SMOKE_SAVE_ARGS=()
fi

OUTPUT="${RDD3D_FM_OUTPUT:-${ROOT}/results_rdd3d/${RUN_NAME}}"
LOG_DIR="${ROOT}/logs"
TRAIN_LOG="${RDD3D_FM_TRAIN_LOG:-${LOG_DIR}/${RUN_NAME}.log}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: missing Python environment: ${PYTHON}" >&2
    exit 2
fi
if [[ ! -f "${TRAIN_ENTRY}" ]]; then
    echo "ERROR: missing training entry: ${TRAIN_ENTRY}" >&2
    exit 2
fi
if [[ ! -d "${TRAIN_LATENTS}" || ! -d "${CONDITIONS}" ]]; then
    echo "ERROR: missing train latents or DINO conditions under ${SPLIT_ROOT}" >&2
    exit 2
fi
if (( GLOBAL_BATCH_SIZE % NPROC != 0 )); then
    echo "ERROR: global batch ${GLOBAL_BATCH_SIZE} must be divisible by ${NPROC} GPUs" >&2
    exit 2
fi

RESUME_ARGS=()
if [[ -n "${RDD3D_FM_RESUME:-}" ]]; then
    [[ -f "${RDD3D_FM_RESUME}" ]] || {
        echo "ERROR: resume checkpoint does not exist: ${RDD3D_FM_RESUME}" >&2
        exit 2
    }
    RESUME_ARGS=(--resume "${RDD3D_FM_RESUME}")
elif [[ -e "${OUTPUT}/checkpoints/latest.pt" ]]; then
    echo "ERROR: refusing to mix a from-scratch run with existing output: ${OUTPUT}" >&2
    echo "Set RDD3D_FM_RESUME explicitly to resume it." >&2
    exit 2
fi

BACKWARD_ARGS=()
if grep -q -- "--backward-per-stage" "${TRAIN_ENTRY}"; then
    BACKWARD_ARGS=(--backward-per-stage)
else
    echo "WARNING: ${TRAIN_ENTRY} lacks --backward-per-stage; peak memory will be higher." >&2
fi

mkdir -p "${OUTPUT}" "${LOG_DIR}"
cd "${ROOT}"
export PYTHONPATH="${TRAIN_MODULE_DIR}:${ROOT}/RDD_DiT_3D:${ROOT}/third_party/TRELLIS.2:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${RDD3D_FM_OMP_THREADS:-2}"

if [[ -z "${MASTER_PORT:-}" ]]; then
    MASTER_PORT="$(${PYTHON} -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
fi

ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / NPROC))
echo "============================================================"
echo "RDD-3D FM Unified XL-scale — 2-D 300/300/400 parity"
echo "Mode:                 $([[ "${SMOKE_TEST}" == "1" ]] && echo smoke || echo production)"
echo "GPUs:                 ${CUDA_VISIBLE_DEVICES} (${NPROC} ranks)"
echo "Model:                width=${MODEL_WIDTH}, depth=${MODEL_DEPTH}, no stage embedding"
echo "FM direction:         noise(t=0) -> data(t=1)"
echo "FM stages:            8^3 [0,.4], 16^3 [.4,.7], 32^3 [.7,1]"
echo "Resolution allocation: fine/middle/coarse = 300/300/400"
echo "Stage training:       all three stages on every shape"
echo "Global shape batch:   ${GLOBAL_BATCH_SIZE} (${ACCUMULATION_STEPS} microsteps)"
echo "Optimizer:            AdamW lr=1e-4, weight_decay=0"
echo "EMA:                  0.9999"
echo "Velocity loss:        0.5 * L_mass + 0.5 * L_feature"
echo "Sparse auxiliary:     0.25 * BCE, pos_weight=1"
echo "Heuristics:           disabled"
echo "Precision:            ${AMP_MODE}"
echo "Optimizer updates:    ${TOTAL_STEPS}"
echo "Train assets:         ${MAX_ASSETS}"
echo "Output:               ${OUTPUT}"
echo "Log:                  ${TRAIN_LOG}"
echo "Training entry:       ${TRAIN_ENTRY}"
echo "============================================================"

set -o pipefail
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON}" -m torch.distributed.run \
    --nnodes=1 --nproc_per_node="${NPROC}" \
    --master_addr=127.0.0.1 --master_port="${MASTER_PORT}" \
    "${TRAIN_ENTRY}" \
    --latent-dir "${TRAIN_LATENTS}" --condition-dir "${CONDITIONS}" \
    --output "${OUTPUT}" "${RESUME_ARGS[@]}" \
    --fine-resolution 32 --resolutions 8 16 32 \
    --time-fractions 0.3 0.3 0.4 \
    --flow-path rdd-fm-global --all-stages-per-step "${BACKWARD_ARGS[@]}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --max-assets "${MAX_ASSETS}" --steps "${TOTAL_STEPS}" \
    "${SMOKE_SAVE_ARGS[@]}" \
    --width "${MODEL_WIDTH}" --depth "${MODEL_DEPTH}" \
    --lr 1e-4 --weight-decay 0 --ema-rate 0.9999 \
    --velocity-loss-mode group-balanced --mass-velocity-weight 1 \
    --occupancy-loss-weight 0.25 --positive-weight 1 \
    --negative-parent-ratio 0 --endpoint-time-probability 0 \
    --generated-parent-probability 0 \
    --generated-endpoint-time-probability 0 \
    --generated-parent-velocity-weight 1 \
    --grad-clip 0 --amp "${AMP_MODE}" --seed 0 \
    --save-every "${SAVE_EVERY}" --latest-every "${LATEST_EVERY}" \
    --log-every "${LOG_EVERY}" \
    2>&1 | tee -a "${TRAIN_LOG}"

echo "RDD3D_FM_PARITY_COMPLETE output=${OUTPUT} steps=${TOTAL_STEPS}"
