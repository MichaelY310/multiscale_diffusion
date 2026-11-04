#!/usr/bin/env bash
set -euo pipefail

# Medium-scale RDD-3D experiment.  This deliberately does not download the
# complete Objaverse-XL/TRELLIS-500K corpus.  It selects a deterministic 1,280
# candidate pool disjoint from every earlier RDD-3D pool, preprocesses TRELLIS
# SLat + DINOv2 conditions, freezes 896/96/96 train/dev/final-test splits, and
# trains a ~94M shared 8^3->16^3->32^3 dynamic-sparse model on four GPUs.
#
# The final-test split is only created here.  This script never evaluates on it.
# checkpoints/latest.pt is refreshed atomically every 100 optimizer steps, so
# the tmux job may be interrupted and restarted safely.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${RDD3D_DYNAMIC_PYTHON:-${ROOT}/.venv_rdd3d/bin/python}"

CANDIDATES="${RDD3D_SCALE_CANDIDATES:-1280}"
TRAIN_COUNT="${RDD3D_SCALE_TRAIN_COUNT:-896}"
DEV_COUNT="${RDD3D_SCALE_DEV_COUNT:-96}"
TEST_COUNT="${RDD3D_SCALE_TEST_COUNT:-96}"
SUBSET_SEED="${RDD3D_SCALE_SUBSET_SEED:-20260813}"
SPLIT_SEED="${RDD3D_SCALE_SPLIT_SEED:-104729}"
SOURCE="${RDD3D_SCALE_SOURCE:-github}"

DATA_ROOT="${RDD3D_SCALE_DATA_ROOT:-${ROOT}/datasets/rdd3d_scale1k/ObjaverseXL_${SOURCE}_train_${CANDIDATES}}"
REPORT_ROOT="${ROOT}/eval_results/rdd3d_scale1k/preprocess_${CANDIDATES}"
SPLIT_ROOT="${DATA_ROOT}/rdd_dynamic_split_${TRAIN_COUNT}train_${DEV_COUNT}dev_${TEST_COUNT}test_seed${SPLIT_SEED}"
LATENTS="${DATA_ROOT}/shape_latents/shape_enc_next_dc_f16c32_fp16_512"
RENDERS="${DATA_ROOT}/renders_cond"
TRAIN_LATENTS="${SPLIT_ROOT}/train_latents"
ALL_LATENTS="${SPLIT_ROOT}/all_latents"
CONDITIONS="${SPLIT_ROOT}/conditions_dinov2_vits14"

WIDTH="${RDD3D_SCALE_WIDTH:-640}"
DEPTH="${RDD3D_SCALE_DEPTH:-16}"
TEACHER_TOTAL="${RDD3D_SCALE_TEACHER_STEPS:-30000}"
ONPOLICY_STEPS="${RDD3D_SCALE_ONPOLICY_STEPS:-70000}"
ONPOLICY_FINAL=$((TEACHER_TOTAL + ONPOLICY_STEPS))
RUN_NAME="${RDD3D_SCALE_RUN_NAME:-dynamic_sparse_scale1k_w${WIDTH}d${DEPTH}_4gpu}"
OUTPUT_ROOT="${ROOT}/results_rdd3d/${RUN_NAME}"
TEACHER_RUN="${OUTPUT_ROOT}/teacher"
ONPOLICY_RUN="${OUTPUT_ROOT}/onpolicy"
LOG_ROOT="${ROOT}/logs/${RUN_NAME}"
EVAL_ROOT="${ROOT}/eval_results/rdd3d_dynamic_sparse/${RUN_NAME}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing Python environment: ${PYTHON}" >&2
    exit 2
fi
mkdir -p "${DATA_ROOT}" "${REPORT_ROOT}" "${OUTPUT_ROOT}" "${LOG_ROOT}" "${EVAL_ROOT}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/RDD_DiT_3D:${ROOT}/third_party/TRELLIS.2:${PYTHONPATH:-}"

echo "===== Phase 0/6: deterministic disjoint ${CANDIDATES}-candidate subset ====="
if [[ ! -f "${DATA_ROOT}/metadata.csv" ]]; then
    SUBSET_ARGS=(
        --output-root "${DATA_ROOT}" --source "${SOURCE}"
        --count "${CANDIDATES}" --seed "${SUBSET_SEED}" --split train
        --min-aesthetic-score 5.5
    )
    for OLD_ASSETS in \
        "${ROOT}/datasets/rdd3d_pilot/ObjaverseXL_github/shape_latents" \
        "${ROOT}/datasets/rdd3d_formal/ObjaverseXL_github_train_256/metadata.csv"; do
        if [[ -e "${OLD_ASSETS}" ]]; then
            SUBSET_ARGS+=(--exclude "${OLD_ASSETS}")
        fi
    done
    "${PYTHON}" scripts/rdd3d/create_objaversexl_subset.py "${SUBSET_ARGS[@]}"
fi

echo "===== Phase 1/6: download and TRELLIS SLat/render preprocessing ====="
RDD3D_SPLIT=train \
RDD3D_SOURCE="${SOURCE}" \
RDD3D_SUBSET_COUNT="${CANDIDATES}" \
RDD3D_SUBSET_SEED="${SUBSET_SEED}" \
RDD3D_OVOXEL_RESOLUTION=512 \
RDD3D_DATA_ROOT="${DATA_ROOT}" \
RDD3D_REPORT_ROOT="${REPORT_ROOT}" \
RDD3D_DOWNLOAD_WORLD_SIZE=1 \
RDD3D_DOWNLOAD_RANK=0 \
RDD3D_GPU=0 \
RDD3D_RENDER_GPUS=0,1,2,3 \
RDD3D_RENDER_COND=1 \
RDD3D_NUM_VIEWS=16 \
RDD3D_MAX_WORKERS=24 \
    bash scripts/rdd3d/prepare_trellis2_formal_split.sh

echo "===== Phase 2/6: frozen train/dev/final-test split and DINOv2 conditions ====="
"${PYTHON}" scripts/rdd3d/build_dynamic_sparse_formal_split.py \
    --latent-dir "${LATENTS}" --render-dir "${RENDERS}" \
    --output-root "${SPLIT_ROOT}" --train-count "${TRAIN_COUNT}" \
    --heldout-count 0 --dev-count "${DEV_COUNT}" --test-count "${TEST_COUNT}" \
    --seed "${SPLIT_SEED}" --fine-resolution 32

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" RDD_DiT_3D/precompute_dynamic_sparse_conditions.py \
    --render-dir "${RENDERS}" --latent-dir "${ALL_LATENTS}" \
    --output "${CONDITIONS}" --model dinov2_vits14

"${PYTHON}" RDD_DiT_3D/analyze_dynamic_sparse_rdd.py \
    --latent-dir "${TRAIN_LATENTS}" --limit "${TRAIN_COUNT}" \
    --fine-resolution 32 --resolutions 8 16 32 --steps 8 12 20 \
    --output "${EVAL_ROOT}/train_oracle.json"

checkpoint_step() {
    "${PYTHON}" -c 'import sys,torch; print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("step", 0)))' "$1"
}

train_to_total() {
    local output="$1"
    local total="$2"
    local fallback_checkpoint="$3"
    shift 3
    local latest="${output}/checkpoints/latest.pt"
    local start=0
    local resume_args=()
    if [[ -e "${latest}" ]]; then
        start="$(checkpoint_step "${latest}")"
        resume_args=(--resume "${latest}" --resume-weights model)
    elif [[ -n "${fallback_checkpoint}" && -e "${fallback_checkpoint}" ]]; then
        start="$(checkpoint_step "${fallback_checkpoint}")"
        resume_args=(--resume "${fallback_checkpoint}" --resume-weights model --reset-optimizer)
    fi
    local remaining=$((total - start))
    if (( remaining <= 0 )); then
        echo "${output}: already reached step ${start}/${total}"
        return 0
    fi
    echo "${output}: training ${remaining} steps from ${start} to ${total}"
    CUDA_VISIBLE_DEVICES=0,1,2,3 "${PYTHON}" -m torch.distributed.run \
        --standalone --nproc_per_node 4 RDD_DiT_3D/train_dynamic_sparse_rdd.py \
        --latent-dir "${TRAIN_LATENTS}" --condition-dir "${CONDITIONS}" \
        --output "${output}" "${resume_args[@]}" \
        --fine-resolution 32 --resolutions 8 16 32 \
        --time-fractions 0.5 0.3 0.2 --max-assets "${TRAIN_COUNT}" \
        --steps "${remaining}" --width "${WIDTH}" --depth "${DEPTH}" "$@" \
        --ema-rate 0.9995 --occupancy-loss-weight 0.75 --positive-weight 3.0 \
        --negative-parent-ratio 1.0 --endpoint-time-probability 0.25 \
        --mass-velocity-weight 4 --save-every 2000 --latest-every 100 \
        --log-every 20 --amp bf16 --seed 20260813
}

echo "===== Phase 3/6: teacher-parent warm-up, ~94M parameters on four GPUs ====="
train_to_total "${TEACHER_RUN}" "${TEACHER_TOTAL}" "" \
    --lr 1e-4 --stage-probabilities 0.25 0.30 0.45

TEACHER_CHECKPOINT="${TEACHER_RUN}/checkpoints/latest.pt"
THRESHOLDS="${EVAL_ROOT}/teacher_step${TEACHER_TOTAL}_clean_thresholds.json"
echo "===== Phase 4/6: four generated-parent replay trajectories ====="
if [[ ! -s "${THRESHOLDS}" ]]; then
    CUDA_VISIBLE_DEVICES=0 "${PYTHON}" RDD_DiT_3D/calibrate_dynamic_sparse_pruning.py \
        --checkpoint "${TEACHER_CHECKPOINT}" --latent-dir "${TRAIN_LATENTS}" \
        --condition-dir "${CONDITIONS}" --output "${THRESHOLDS}" \
        --target-recall 0.999 0.995 0.99 --limit "${TRAIN_COUNT}" --weights model
fi

REPLAY_DIRS=()
REPLAY_PIDS=()
for GPU_SEED in 0:424242 1:777777 2:13579 3:24680; do
    GPU="${GPU_SEED%%:*}"
    SEED="${GPU_SEED##*:}"
    REPLAY="${OUTPUT_ROOT}/replay_seed${SEED}"
    REPLAY_DIRS+=("${REPLAY}")
    mkdir -p "${REPLAY}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
        --checkpoint "${TEACHER_CHECKPOINT}" --latent-dir "${TRAIN_LATENTS}" \
        --condition-dir "${CONDITIONS}" --thresholds "${THRESHOLDS}" \
        --output-dir "${REPLAY}" --steps 8 12 20 \
        --mass-thresholds 0.05 0.10 0.25 --limit "${TRAIN_COUNT}" \
        --seed "${SEED}" --weights model \
        >"${LOG_ROOT}/replay_seed${SEED}.log" 2>&1 &
    REPLAY_PIDS+=("$!")
done
for PID in "${REPLAY_PIDS[@]}"; do
    wait "${PID}"
done

echo "===== Phase 5/6: on-policy generated-parent training ====="
train_to_total "${ONPOLICY_RUN}" "${ONPOLICY_FINAL}" "${TEACHER_CHECKPOINT}" \
    --lr 3e-5 --stage-probabilities 0.25 0.30 0.45 \
    --generated-parent-dirs "${REPLAY_DIRS[@]}" \
    --generated-parent-probability 0.75 \
    --generated-endpoint-time-probability 0.50 \
    --generated-parent-velocity-weight 0.75

echo "===== Phase 6/6: complete (final test remains untouched) ====="
echo "Data manifest: ${DATA_ROOT}/rdd3d_subset_manifest.json"
echo "Split manifest: ${SPLIT_ROOT}/manifest.json"
echo "Teacher latest: ${TEACHER_RUN}/checkpoints/latest.pt"
echo "On-policy latest: ${ONPOLICY_RUN}/checkpoints/latest.pt"
echo "RDD3D_SCALE1K_COMPLETE"
