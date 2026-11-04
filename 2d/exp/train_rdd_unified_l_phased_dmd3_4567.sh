#!/usr/bin/env bash
# RDD-Phased-DMD: exactly one conditional call at 8x8, 16x16 and 32x32.
# The visible-device list is intentionally fixed to physical GPUs 4,5,6,7.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${REPO_ROOT}/checkpoints/RDD-Unified-L/0400000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results_rdd_dmd3/unified_l_300300400_400k_phased_dmd3_exact}"
MAX_STEPS="${MAX_STEPS:-30000}"
BATCH_PER_GPU="${BATCH_PER_GPU:-8}"
CRITIC_UPDATES="${CRITIC_UPDATES:-5}"
CRITIC_WARMUP_STEPS="${CRITIC_WARMUP_STEPS:-100}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-10}"
MASTER_PORT="${MASTER_PORT:-29637}"
NO_SAVE="${NO_SAVE:-0}"

# NCCL 2.26 on this eight-card Blackwell host intermittently raises CUDA 700
# while reducing DiT-L gradients over direct PCIe P2P.  Shared-memory transport
# is stable and leaves the mathematical/global-batch semantics unchanged.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

cd "${REPO_ROOT}"
[[ -s "${TEACHER_CHECKPOINT}" ]] || {
    echo "ERROR: missing RDD teacher: ${TEACHER_CHECKPOINT}" >&2
    exit 1
}

if [[ -f "${REPO_ROOT}/.venv-rdd-eval/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv-rdd-eval/bin/activate"
elif [[ -f "${REPO_ROOT}/2d/2d/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/2d/2d/bin/activate"
else
    echo "ERROR: RDD Python environment not found" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}/.control"
token="${OUTPUT_DIR}/.control/stop-$(date -u +%Y%m%dT%H%M%SZ)-$$"
record="${OUTPUT_DIR}/.control/process-$(date -u +%Y%m%dT%H%M%SZ)-$$.txt"

request_stop() {
    printf 'launcher received TERM/INT at %s\n' "$(date -Is)" > "${token}"
}
trap request_stop TERM INT

resume_args=()
if [[ -s "${OUTPUT_DIR}/checkpoints/latest.pt" ]]; then
    resume_args=(--resume "${OUTPUT_DIR}/checkpoints/latest.pt")
fi
save_args=()
if [[ "${NO_SAVE}" == "1" ]]; then
    save_args=(--no-save)
fi

echo "============================================================"
echo "RDD-Phased-DMD-3NFE"
echo "GPUs: physical 4,5,6,7; batch/GPU=${BATCH_PER_GPU}; global batch=$((4 * BATCH_PER_GPU))"
echo "Teacher: ${TEACHER_CHECKPOINT}"
echo "Schedule: one call each at 8x8, 16x16, 32x32"
echo "DMD: exact RDD A_t^T A_t correction; critic:student=${CRITIC_UPDATES}:1"
echo "Output: ${OUTPUT_DIR}"
echo "Stop token: ${token}"
echo "============================================================"

setsid env CUDA_VISIBLE_DEVICES=4,5,6,7 \
    torchrun --standalone --master_port "${MASTER_PORT}" --nproc_per_node=4 \
    2d/train_rdd_phased_dmd3.py \
        --teacher-checkpoint "${TEACHER_CHECKPOINT}" \
        --output-dir "${OUTPUT_DIR}" \
        --model DiT-L/2 \
        --resolutions 32,16,8 \
        --stage-lengths 300,300,400 \
        --max-steps "${MAX_STEPS}" \
        --batch-per-gpu "${BATCH_PER_GPU}" \
        --critic-updates "${CRITIC_UPDATES}" \
        --critic-warmup-steps "${CRITIC_WARMUP_STEPS}" \
        --teacher-cfg 1.5 \
        --student-lr 1e-5 \
        --critic-lr 1e-5 \
        --preservation-weight 0.05 \
        --path-correction exact \
        --checkpoint-every "${CHECKPOINT_EVERY}" \
        --log-every "${LOG_EVERY}" \
        --stop-file "${token}" \
        --bf16 \
        "${resume_args[@]}" "${save_args[@]}" &
torchrun_pid=$!
sleep 1
torchrun_pgid="$(ps -o pgid= -p "${torchrun_pid}" | tr -d ' ')"
{
    printf 'pid=%s\n' "${torchrun_pid}"
    printf 'pgid=%s\n' "${torchrun_pgid}"
    printf 'uid=%s\n' "$(id -u)"
    printf 'stop_file=%s\n' "${token}"
    printf 'gpus=4,5,6,7\n'
    printf 'output_dir=%s\n' "${OUTPUT_DIR}"
} > "${record}"

set +e
wait "${torchrun_pid}"
status=$?
set -e
echo "torchrun exit status: ${status}"
exit "${status}"
