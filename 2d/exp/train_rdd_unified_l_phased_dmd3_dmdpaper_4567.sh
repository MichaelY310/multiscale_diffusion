#!/usr/bin/env bash
# Three-step Unified-L RDD distribution-matching distillation on GPUs 4-7.
#
# RDD parity:
#   DiT-L/2, 300/300/400, effective global batch 256, BF16, seed 0.
# Original DMD ImageNet recipe where applicable:
#   AdamW 2e-6, betas (0.9, 0.999), wd 0.01, 500-step warmup,
#   one generator update and one fake-score update per outer step,
#   score timesteps [0.02T, 0.98T], grad clip 10, 350K optimizer steps.
#
# The original DMD ImageNet experiment used an EDM/ADM denoiser, not DiT. This
# launcher transfers its optimizer/loss recipe to the pretrained RDD DiT-L/2.
# The RDD score and generator gradient are obtained by substituting this repo's
# q_sample expression directly into DMD Eq. 7 and Eq. 8.  No DMD2 TTUR/GAN,
# endpoint preservation, selectable path correction, or gradient-value clip is
# present.
#
# At the scheduled America/Los_Angeles stop time, a job-scoped watchdog asks
# the trainer to save an exact-step checkpoint and exit cleanly.  A caller may
# pass STOP_AT_EPOCH; otherwise the next 09:00 is used.  If saving stalls, only
# same-UID PIDs in this run's private process group can be signalled.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${REPO_ROOT}/checkpoints/RDD-Unified-L/0400000.pt}"
REGRESSION_DATA="${REGRESSION_DATA:-${REPO_ROOT}/data/rdd_dmd_pairs/unified_l_300300400_cfg1.5_marginalddim250_v2}"
VAE_PATH="${VAE_PATH:-${REPO_ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results_rdd_dmd3/unified_l_300300400_400k_rdd_dmd_full_v2_gb256}"
MAX_STEPS="${MAX_STEPS:-350000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
MICRO_BATCH_PER_GPU="${MICRO_BATCH_PER_GPU:-4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"
LOG_EVERY="${LOG_EVERY:-10}"
MASTER_PORT="${MASTER_PORT:-29639}"
NO_SAVE="${NO_SAVE:-0}"
SAVE_GRACE_SECONDS="${SAVE_GRACE_SECONDS:-300}"
TERM_GRACE_SECONDS="${TERM_GRACE_SECONDS:-10}"
STOP_AT_EPOCH="${STOP_AT_EPOCH:-}"
SF_TIMEZONE="America/Los_Angeles"
GPU_LIST="4,5,6,7"
WORLD_SIZE=4

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

cd "${REPO_ROOT}"
[[ -s "${TEACHER_CHECKPOINT}" ]] || {
    echo "ERROR: missing RDD teacher: ${TEACHER_CHECKPOINT}" >&2
    exit 1
}
[[ -s "${REGRESSION_DATA}/metadata.json" ]] || {
    echo "ERROR: first build the original-DMD regression pairs: ${REGRESSION_DATA}" >&2
    exit 1
}
[[ -s "${VAE_PATH}/config.json" ]] || {
    echo "ERROR: missing local SD VAE: ${VAE_PATH}" >&2
    exit 1
}
command -v flock >/dev/null 2>&1 || {
    echo "ERROR: flock is required for single-launch ownership" >&2
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

if (( GLOBAL_BATCH_SIZE <= 0 || MICRO_BATCH_PER_GPU <= 0 )); then
    echo "ERROR: batch sizes must be positive" >&2
    exit 1
fi
GLOBAL_MICRO_BATCH=$((WORLD_SIZE * MICRO_BATCH_PER_GPU))
if (( GLOBAL_BATCH_SIZE % GLOBAL_MICRO_BATCH != 0 )); then
    echo "ERROR: global batch ${GLOBAL_BATCH_SIZE} must be divisible by ${WORLD_SIZE} GPUs x micro ${MICRO_BATCH_PER_GPU}" >&2
    exit 1
fi
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / GLOBAL_MICRO_BATCH))

mkdir -p "${OUTPUT_DIR}/.control" "${REPO_ROOT}/logs"
exec 9>"${OUTPUT_DIR}/.control/launcher.lock"
if ! flock -n 9; then
    echo "ERROR: another launcher owns ${OUTPUT_DIR}/.control/launcher.lock" >&2
    exit 1
fi

run_token="$(date -u +%Y%m%dT%H%M%SZ)-$$"
stop_file="${OUTPUT_DIR}/.control/stop-${run_token}"
checkpoint_request_file="${OUTPUT_DIR}/.control/checkpoint-${run_token}"
pid_record="${OUTPUT_DIR}/.control/process-${run_token}.txt"
watchdog_log_file="${REPO_ROOT}/logs/rdd_unified_l_300300400_rdd_dmd_full_v2_gb256_la9_watchdog.log"

resume_args=()
if [[ -s "${OUTPUT_DIR}/checkpoints/latest.pt" ]]; then
    resume_args=(--resume "${OUTPUT_DIR}/checkpoints/latest.pt")
fi
no_save_args=()
if [[ "${NO_SAVE}" == "1" ]]; then
    no_save_args=(--no-save)
fi

watchdog_log() {
    printf '[%s] %s\n' "$(date -Is)" "$1" | tee -a "${watchdog_log_file}"
}

owned_group_pids() {
    local process_group="$1"
    ps -eo pid=,pgid=,uid= | awk \
        -v wanted_pgid="${process_group}" -v wanted_uid="${UID}" \
        '$2 == wanted_pgid && $3 == wanted_uid { print $1 }'
}

group_is_alive() {
    [[ -n "$(owned_group_pids "$1")" ]]
}

signal_owned_group() {
    local signal_name="$1"
    local process_group="$2"
    local process_ids=()
    mapfile -t process_ids < <(owned_group_pids "${process_group}")
    for process_id in "${process_ids[@]}"; do
        kill "-${signal_name}" "${process_id}" 2>/dev/null || true
    done
}

resolve_stop_schedule() {
    python3 - "${STOP_AT_EPOCH}" <<'PY'
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

zone = ZoneInfo("America/Los_Angeles")
now = datetime.now(zone)
explicit_epoch = sys.argv[1]
if explicit_epoch:
    try:
        target = datetime.fromtimestamp(int(explicit_epoch), zone)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SystemExit(f"invalid STOP_AT_EPOCH={explicit_epoch!r}: {exc}")
else:
    target = now.replace(hour=9, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
print(max(0, int((target - now).total_seconds())), target.isoformat())
PY
}

scheduled_stop_watchdog() {
    local process_group="$1"
    local delay_seconds target_time
    read -r delay_seconds target_time < <(resolve_stop_schedule)
    watchdog_log "Watching private PGID ${process_group}; exact-save stop is ${target_time}."
    sleep "${delay_seconds}"
    if ! group_is_alive "${process_group}"; then
        watchdog_log "Training exited before the scheduled stop."
        return 0
    fi

    printf 'requested_at=%s\ntimezone=%s\npgid=%s\ntarget=%s\n' \
        "$(date -Is)" "${SF_TIMEZONE}" "${process_group}" "${target_time}" > "${stop_file}"
    watchdog_log "Scheduled stop reached; requested an exact-step checkpoint and clean exit through ${stop_file}."
    local deadline=$((SECONDS + SAVE_GRACE_SECONDS))
    while group_is_alive "${process_group}" && (( SECONDS < deadline )); do sleep 1; done
    if ! group_is_alive "${process_group}"; then
        watchdog_log "Training stopped cleanly after saving the exact checkpoint."
        return 0
    fi

    watchdog_log "Save grace expired; TERM only UID ${UID}, private PGID ${process_group}."
    signal_owned_group TERM "${process_group}"
    deadline=$((SECONDS + TERM_GRACE_SECONDS))
    while group_is_alive "${process_group}" && (( SECONDS < deadline )); do sleep 1; done
    if group_is_alive "${process_group}"; then
        watchdog_log "TERM grace expired; KILL only UID ${UID}, private PGID ${process_group}."
        signal_owned_group KILL "${process_group}"
        sleep 1
        signal_owned_group KILL "${process_group}"
    fi
}

echo "============================================================"
echo "RDD-DMD-3NFE v2 — final-output distribution matching through the full rollout"
echo "GPUs: physical ${GPU_LIST}; per-GPU micro=${MICRO_BATCH_PER_GPU}; accumulation=${GRADIENT_ACCUMULATION_STEPS}; effective global batch=${GLOBAL_BATCH_SIZE}"
echo "Model=DiT-L/2; stages=300/300/400; seed=0; BF16"
echo "LR=2e-6; AdamW betas=(0.9,0.999); wd=0.01; warmup=500; grad_clip=10; max_steps=${MAX_STEPS}"
echo "DMD/fake-score weights=1/1; updates=1:1; global score t uniform on [0.02T,0.98T)"
echo "No endpoint anchor, path switch, DMD2 TTUR, GAN, or gradient-value clipping"
if [[ -n "${STOP_AT_EPOCH}" ]]; then
    echo "LA schedule: exact-save stop at explicit epoch ${STOP_AT_EPOCH}"
else
    echo "LA schedule: exact-save stop at next 09:00 America/Los_Angeles"
fi
echo "Output: ${OUTPUT_DIR}"
if (( ${#resume_args[@]} > 0 )); then
    echo "Resume: ${resume_args[1]}"
else
    echo "Resume: none (fresh run)"
fi
echo "============================================================"

setsid env CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
    torchrun --standalone --master_port "${MASTER_PORT}" --nproc_per_node="${WORLD_SIZE}" \
    2d/train_rdd_phased_dmd3.py \
        --teacher-checkpoint "${TEACHER_CHECKPOINT}" \
        --regression-data "${REGRESSION_DATA}" \
        --vae-path "${VAE_PATH}" \
        --output-dir "${OUTPUT_DIR}" \
        --model DiT-L/2 \
        --resolutions 32,16,8 \
        --stage-lengths 300,300,400 \
        --max-steps "${MAX_STEPS}" \
        --batch-per-gpu "${MICRO_BATCH_PER_GPU}" \
        --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --teacher-cfg 1.5 \
        --student-lr 2e-6 \
        --critic-lr 2e-6 \
        --weight-decay 0.01 \
        --student-beta1 0.9 --student-beta2 0.999 \
        --critic-beta1 0.9 --critic-beta2 0.999 \
        --lr-warmup-steps 500 \
        --max-grad-norm 10 \
        --checkpoint-every "${CHECKPOINT_EVERY}" \
        --log-every "${LOG_EVERY}" \
        --seed 0 \
        --checkpoint-request-file "${checkpoint_request_file}" \
        --stop-file "${stop_file}" \
        --bf16 \
        "${no_save_args[@]}" \
        "${resume_args[@]}" &
train_pid=$!
sleep 1
train_pgid="$(ps -o pgid= -p "${train_pid}" | awk '{print $1}')"
if [[ -z "${train_pgid}" || "${train_pgid}" != "${train_pid}" ]]; then
    echo "ERROR: failed to create private process group (pid=${train_pid}, pgid=${train_pgid:-missing})" >&2
    kill "${train_pid}" 2>/dev/null || true
    wait "${train_pid}" 2>/dev/null || true
    exit 1
fi
printf 'pid=%s\npgid=%s\nuid=%s\ncheckpoint_request_file=%s\nstop_file=%s\ngpus=%s\noutput_dir=%s\n' \
    "${train_pid}" "${train_pgid}" "${UID}" "${checkpoint_request_file}" "${stop_file}" "${GPU_LIST}" "${OUTPUT_DIR}" > "${pid_record}"

watchdog_pid=""
cleanup_launcher() {
    local exit_status=$?
    trap - EXIT HUP INT TERM
    if [[ -n "${watchdog_pid}" ]] && kill -0 "${watchdog_pid}" 2>/dev/null; then
        kill "${watchdog_pid}" 2>/dev/null || true
        wait "${watchdog_pid}" 2>/dev/null || true
    fi
    if [[ -n "${train_pid}" ]] && kill -0 "${train_pid}" 2>/dev/null; then
        printf 'launcher_exit_at=%s\n' "$(date -Is)" > "${stop_file}"
        signal_owned_group TERM "${train_pgid}"
    fi
    rm -f -- "${pid_record}"
    exit "${exit_status}"
}
trap cleanup_launcher EXIT HUP INT TERM

scheduled_stop_watchdog "${train_pgid}" &
watchdog_pid=$!

set +e
wait "${train_pid}"
train_status=$?
set -e
train_pid=""

if [[ -f "${stop_file}" ]]; then
    watchdog_log "Launcher observed scheduled/scoped stop; torchrun exit status=${train_status}."
    exit 0
fi
if (( train_status != 0 )); then
    echo "ERROR: torchrun exited with status ${train_status}" >&2
    exit "${train_status}"
fi
echo "Training completed normally."
