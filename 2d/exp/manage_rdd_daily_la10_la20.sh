#!/usr/bin/env bash
# Recurring LA 09:00/20:00 cycle (legacy filename retained for compatibility).
# Current jobs are stopped cleanly at the next 09:00; a preview is then sampled.
# Every day at 20:00 the two jobs resume, with their own scoped watchdogs set to
# save and stop at the following 09:00.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
LOG_FILE="${REPO_ROOT}/logs/manage_rdd_daily_la9_la20.log"
TIMEZONE="America/Los_Angeles"
CONTROL_500="${REPO_ROOT}/checkpoints/RDD-Unified-L-2Stage-500-500/.control"
CONTROL_DMD="${REPO_ROOT}/results_rdd_dmd3/unified_l_300300400_400k_rdd_dmd_full_v2_gb256/.control"
SESSION_500="rdd_unified_l_2stage_500500"
SESSION_DMD="rdd_unified_l_phased_dmd3_gb256"
SESSION_SAMPLE="rdd_dmd_latest_preview_after_la9"
OLD_WATCHDOG_SLEEP_PIDS="${OLD_WATCHDOG_SLEEP_PIDS:-}"

mkdir -p "${REPO_ROOT}/logs"

log() {
    printf '[%s] %s\n' "$(date -Is)" "$1" | tee -a "${LOG_FILE}"
}

next_la_epoch() {
    local hour="$1"
    python3 - "${hour}" <<'PY'
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
zone = ZoneInfo("America/Los_Angeles")
now = datetime.now(zone)
target = now.replace(hour=int(sys.argv[1]), minute=0, second=0, microsecond=0)
if target <= now:
    target += timedelta(days=1)
print(int(target.timestamp()), target.isoformat())
PY
}

sleep_until_epoch() {
    local epoch="$1"
    local delay=$((epoch - $(date +%s)))
    (( delay <= 0 )) || sleep "${delay}"
}

latest_process_record() {
    local control_dir="$1"
    find "${control_dir}" -maxdepth 1 -type f -name 'process-*.txt' -printf '%T@ %p\n' 2>/dev/null \
        | sort -n | tail -n 1 | cut -d' ' -f2-
}

record_value() {
    local record="$1"
    local key="$2"
    awk -F= -v key="${key}" '$1 == key {sub(/^[^=]*=/, ""); print; exit}' "${record}"
}

owned_group_is_alive() {
    local pgid="$1"
    ps -eo pgid=,uid= | awk -v pgid="${pgid}" -v uid="${UID}" '$1 == pgid && $2 == uid {found=1} END {exit !found}'
}

signal_owned_group() {
    local signal_name="$1"
    local pgid="$2"
    local pid
    while read -r pid; do
        [[ -z "${pid}" ]] || kill "-${signal_name}" "${pid}" 2>/dev/null || true
    done < <(ps -eo pid=,pgid=,uid= | awk -v pgid="${pgid}" -v uid="${UID}" '$2 == pgid && $3 == uid {print $1}')
}

release_old_watchdog_sleeps() {
    local pid pid_uid command_name
    local normalized_pids="${OLD_WATCHDOG_SLEEP_PIDS//,/ }"
    for pid in ${normalized_pids}; do
        [[ "${pid}" =~ ^[0-9]+$ ]] || continue
        pid_uid="$(ps -o uid= -p "${pid}" 2>/dev/null | tr -d ' ' || true)"
        command_name="$(ps -o comm= -p "${pid}" 2>/dev/null | tr -d ' ' || true)"
        if [[ "${pid_uid}" == "${UID}" && "${command_name}" == "sleep" ]]; then
            kill -CONT "${pid}" 2>/dev/null || true
            log "Released old suspended watchdog sleep PID ${pid}; its trainer is already stopped."
        elif [[ -n "${pid_uid}" ]]; then
            log "Refused to CONT PID ${pid}: it is not this UID's watchdog sleep."
        fi
    done
}

request_clean_stop() {
    local label="$1"
    local control_dir="$2"
    local expected_pattern="$3"
    local record pid pgid record_uid stop_file
    record="$(latest_process_record "${control_dir}")"
    if [[ -z "${record}" || ! -s "${record}" ]]; then
        log "${label}: no live process record; nothing to stop."
        return 0
    fi
    pid="$(record_value "${record}" pid)"
    pgid="$(record_value "${record}" pgid)"
    record_uid="$(record_value "${record}" uid)"
    stop_file="$(record_value "${record}" stop_file)"
    if [[ ! "${pid}" =~ ^[0-9]+$ || ! "${pgid}" =~ ^[0-9]+$ || "${record_uid}" != "${UID}" ]]; then
        log "ERROR: ${label} rejected malformed/foreign process record ${record}."
        return 1
    fi
    if ! ps -eo pgid=,uid=,args= | awk -v pgid="${pgid}" -v uid="${UID}" -v pattern="${expected_pattern}" \
        '$1 == pgid && $2 == uid && index($0, pattern) {found=1} END {exit !found}'; then
        log "ERROR: ${label} record does not own the expected training command; refusing to signal or write."
        return 1
    fi
    printf 'requested_at=%s\ntimezone=%s\npgid=%s\n' "$(date -Is)" "${TIMEZONE}" "${pgid}" > "${stop_file}"
    log "${label}: requested exact-save stop through ${stop_file}."
    local deadline=$((SECONDS + 600))
    while owned_group_is_alive "${pgid}" && (( SECONDS < deadline )); do sleep 2; done
    if owned_group_is_alive "${pgid}"; then
        log "${label}: save grace expired; TERM only UID ${UID}, exact PGID ${pgid}."
        signal_owned_group TERM "${pgid}"
        sleep 20
    fi
    if owned_group_is_alive "${pgid}"; then
        log "${label}: TERM grace expired; KILL only UID ${UID}, exact PGID ${pgid}."
        signal_owned_group KILL "${pgid}"
    else
        log "${label}: stopped cleanly."
    fi
}

gpu_group_is_idle() {
    local gpu
    for gpu in "$@"; do
        if nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
            return 1
        fi
    done
    return 0
}

wait_for_gpu_group() {
    local label="$1"
    shift
    until gpu_group_is_idle "$@"; do
        log "${label} is occupied; leaving all existing processes untouched and retrying in 60s."
        sleep 60
    done
}

remove_stale_session() {
    local session="$1"
    local pattern="$2"
    if tmux has-session -t "${session}" 2>/dev/null; then
        if pgrep -u "${UID}" -f "${pattern}" >/dev/null; then
            return 1
        fi
        tmux kill-session -t "${session}"
        log "Removed stale shell-only tmux session ${session}; no matching trainer was alive."
    fi
}

sample_latest_once() {
    local marker="${CONTROL_DMD}/latest_preview_after_la9.done"
    [[ ! -f "${marker}" ]] || { log "Latest preview marker exists; not sampling twice."; return 0; }
    wait_for_gpu_group "GPU 4 for latest preview" 4
    remove_stale_session "${SESSION_SAMPLE}" 'sample_rdd_phased_dmd3.py' || {
        log "Latest preview is already running; leaving it untouched."
        return 0
    }
    local run_tag
    run_tag="$(TZ=${TIMEZONE} date +%Y%m%d_la%H%M%S)"
    tmux new-session -d -s "${SESSION_SAMPLE}" \
        "cd '${REPO_ROOT}' && RUN_TAG='${run_tag}' GPU=4 TOTAL_SAMPLES=16 bash 2d/exp/sample_latest_rdd_dmd_preview.sh && touch '${marker}'"
    log "Started 16-image latest RDD-DMD preview in tmux ${SESSION_SAMPLE}; tag=${run_tag}."
}

start_500500() {
    local stop_epoch="$1"
    [[ ! -s "${REPO_ROOT}/checkpoints/RDD-Unified-L-2Stage-500-500/imagenet256-unified/checkpoints/0400000.pt" ]] || {
        log "500/500 already reached 400K; not restarting it."
        return 0
    }
    if pgrep -u "${UID}" -f 'train_RDD_imagenet256_unified.py.*--stage-lengths 500,500' >/dev/null; then
        log "500/500 is already running; refusing to launch a duplicate."
        return 0
    fi
    wait_for_gpu_group "GPUs 0-3" 0 1 2 3
    remove_stale_session "${SESSION_500}" 'train_RDD_imagenet256_unified.py.*--stage-lengths 500,500' || return 0
    tmux new-session -d -s "${SESSION_500}" \
        "cd '${REPO_ROOT}' && STOP_AT_EPOCH='${stop_epoch}' bash 2d/train_RDD_unified_L_2stage_500_500_4gpu_la9.sh"
    log "Started 500/500 with next LA09 stop epoch ${stop_epoch}."
}

start_dmd() {
    local stop_epoch="$1"
    [[ ! -s "${REPO_ROOT}/results_rdd_dmd3/unified_l_300300400_400k_rdd_dmd_full_v2_gb256/checkpoints/0350000.pt" ]] || {
        log "RDD-DMD already reached 350K; not restarting it."
        return 0
    }
    if pgrep -u "${UID}" -f '2d/train_rdd_phased_dmd3.py' >/dev/null; then
        log "RDD-DMD is already running; refusing to launch a duplicate."
        return 0
    fi
    wait_for_gpu_group "GPUs 4-7" 4 5 6 7
    remove_stale_session "${SESSION_DMD}" '2d/train_rdd_phased_dmd3.py' || return 0
    tmux new-session -d -s "${SESSION_DMD}" \
        "cd '${REPO_ROOT}' && STOP_AT_EPOCH='${stop_epoch}' bash 2d/exp/run_rdd_unified_l_phased_dmd3_dmdpaper_tmux_payload.sh"
    log "Started exact RDD-DMD with next LA09 stop epoch ${stop_epoch}."
}

# Apply the next LA09 stop to jobs that are already running.
read -r first_stop_epoch first_stop_time < <(next_la_epoch 9)
log "Current jobs will stop at ${first_stop_time}; sleeping until then."
sleep_until_epoch "${first_stop_epoch}"
request_clean_stop "500/500" "${CONTROL_500}" 'train_RDD_imagenet256_unified.py' &
first_stop_500_pid=$!
request_clean_stop "RDD-DMD" "${CONTROL_DMD}" 'train_rdd_phased_dmd3.py' &
first_stop_dmd_pid=$!
wait "${first_stop_500_pid}"
wait "${first_stop_dmd_pid}"
release_old_watchdog_sleeps
sample_latest_once

while true; do
    read -r resume_epoch resume_time < <(next_la_epoch 20)
    log "Next resume is ${resume_time}; sleeping on the server."
    sleep_until_epoch "${resume_epoch}"
    read -r stop_epoch stop_time < <(next_la_epoch 9)
    log "LA20 reached; next exact-save pause will be ${stop_time}."
    start_500500 "${stop_epoch}" &
    pid_500=$!
    start_dmd "${stop_epoch}" &
    pid_dmd=$!
    wait "${pid_500}"
    wait "${pid_dmd}"
done
