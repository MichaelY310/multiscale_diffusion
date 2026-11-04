#!/usr/bin/env bash
set -euo pipefail

# Continue the clean 30K teacher-forced scale1k checkpoint to 100K without
# generated-parent replay.  Refinement stages always use ground-truth parents.
# The run is deliberately isolated from the historical on-policy experiment.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${RDD3D_DYNAMIC_PYTHON:-${ROOT}/.venv_rdd3d/bin/python}"

TRAIN_COUNT="${RDD3D_TEACHER_ONLY_TRAIN_COUNT:-896}"
DATA_ROOT="${RDD3D_TEACHER_ONLY_DATA_ROOT:-${ROOT}/datasets/rdd3d_scale1k/ObjaverseXL_github_train_1280}"
SPLIT_ROOT="${RDD3D_TEACHER_ONLY_SPLIT_ROOT:-${DATA_ROOT}/rdd_dynamic_split_896train_96dev_96test_seed104729}"
TRAIN_LATENTS="${SPLIT_ROOT}/train_latents"
CONDITIONS="${SPLIT_ROOT}/conditions_dinov2_vits14"

BASE_CHECKPOINT="${RDD3D_TEACHER_ONLY_BASE_CHECKPOINT:-${ROOT}/results_rdd3d/dynamic_sparse_scale1k_w640d16_4gpu/teacher/checkpoints/latest.pt}"
OUTPUT="${RDD3D_TEACHER_ONLY_OUTPUT:-${ROOT}/results_rdd3d/dynamic_sparse_scale1k_w640d16_teacheronly_8gpu}"
TOTAL_STEPS="${RDD3D_TEACHER_ONLY_TOTAL_STEPS:-100000}"
MASTER_PORT="${RDD3D_TEACHER_ONLY_MASTER_PORT:-29631}"
NPROC="${RDD3D_TEACHER_ONLY_NPROC:-8}"
VISIBLE_GPUS="${RDD3D_TEACHER_ONLY_GPUS:-0,1,2,3,4,5,6,7}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing Python environment: ${PYTHON}" >&2
    exit 2
fi
if [[ ! -e "${BASE_CHECKPOINT}" ]]; then
    echo "Missing clean teacher checkpoint: ${BASE_CHECKPOINT}" >&2
    exit 2
fi
if [[ ! -d "${TRAIN_LATENTS}" || ! -d "${CONDITIONS}" ]]; then
    echo "Missing scale1k train split or DINO conditions under: ${SPLIT_ROOT}" >&2
    exit 2
fi

mkdir -p "${OUTPUT}" "${ROOT}/logs"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/RDD_DiT_3D:${ROOT}/third_party/TRELLIS.2:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${RDD3D_TEACHER_ONLY_OMP_THREADS:-2}"

RESUME="${BASE_CHECKPOINT}"
if [[ -e "${OUTPUT}/checkpoints/latest.pt" ]]; then
    RESUME="${OUTPUT}/checkpoints/latest.pt"
fi

START_STEP="$(${PYTHON} -c 'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("step", 0)))' "${RESUME}")"
REMAINING=$((TOTAL_STEPS - START_STEP))
if (( REMAINING <= 0 )); then
    echo "Teacher-only run is already complete at step ${START_STEP}."
    exit 0
fi

echo "============================================================"
echo "RDD3D scale1k teacher-only continuation"
echo "Resume:       ${RESUME} (step ${START_STEP})"
echo "Output:       ${OUTPUT}"
echo "Target:       ${TOTAL_STEPS} (${REMAINING} remaining steps)"
echo "GPUs:         ${VISIBLE_GPUS} (${NPROC} DDP ranks)"
echo "Parent input: 100% ground truth at 16^3 and 32^3"
echo "Replay:       disabled (generated-parent-probability=0)"
echo "============================================================"

CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}" "${PYTHON}" -m torch.distributed.run \
    --nnodes=1 --nproc_per_node="${NPROC}" \
    --master_addr=127.0.0.1 --master_port="${MASTER_PORT}" \
    RDD_DiT_3D/train_dynamic_sparse_rdd.py \
    --latent-dir "${TRAIN_LATENTS}" --condition-dir "${CONDITIONS}" \
    --output "${OUTPUT}" --resume "${RESUME}" --resume-weights model \
    --fine-resolution 32 --resolutions 8 16 32 \
    --time-fractions 0.5 0.3 0.2 --max-assets "${TRAIN_COUNT}" \
    --steps "${REMAINING}" --width 640 --depth 16 \
    --lr 1e-4 --weight-decay 0.01 --ema-rate 0.9995 \
    --stage-probabilities 0.25 0.30 0.45 \
    --generated-parent-probability 0 \
    --occupancy-loss-weight 0.75 --positive-weight 3.0 \
    --negative-parent-ratio 1.0 --endpoint-time-probability 0.25 \
    --mass-velocity-weight 4 --grad-clip 1.0 \
    --save-every 2000 --latest-every 100 --log-every 20 \
    --amp bf16 --seed 20260813

echo "RDD3D_TEACHER_ONLY_COMPLETE step=${TOTAL_STEPS} output=${OUTPUT}"
