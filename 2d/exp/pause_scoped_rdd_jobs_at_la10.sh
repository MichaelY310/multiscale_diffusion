#!/usr/bin/env bash
# One-shot safety watchdog for exactly one sampler PGID and one RDD trainer PGID.
# It never scans GPUs, process names globally, or signals processes outside the
# two caller-provided private groups owned by the current UID.

set -Eeuo pipefail

: "${SAMPLE_PGID:?set the private PGID of the sampler launcher}"
: "${SAMPLE_MARKER:?set an exact command-line marker for the sampler}"
: "${TRAIN_PGID:?set the private PGID of the trainer launcher}"
: "${TRAIN_MARKER:?set an exact command-line marker for the trainer}"
: "${TRAIN_STOP_FILE:?set the unique --stop-file used by the trainer}"

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/logs/pause_scoped_rdd_jobs_at_la10.log}"
TRAIN_SAVE_GRACE_SECONDS="${TRAIN_SAVE_GRACE_SECONDS:-300}"
SAMPLE_TERM_GRACE_SECONDS="${SAMPLE_TERM_GRACE_SECONDS:-30}"
mkdir -p "$(dirname "${LOG_FILE}")"

log() {
    printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${LOG_FILE}"
}

for value in "${SAMPLE_PGID}" "${TRAIN_PGID}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
        echo "ERROR: invalid PGID ${value}" >&2
        exit 1
    }
done

expected_control="${REPO_ROOT}/checkpoints/RDD-Unified-L-400-300-300/.control"
case "${TRAIN_STOP_FILE}" in
    "${expected_control}"/stop-*) ;;
    *)
        echo "ERROR: refusing stop file outside ${expected_control}: ${TRAIN_STOP_FILE}" >&2
        exit 1
        ;;
esac

owned_group_rows() {
    local wanted_pgid="$1"
    ps -eo pid=,pgid=,uid=,args= | awk \
        -v wanted_pgid="${wanted_pgid}" -v wanted_uid="${UID}" \
        '$2 == wanted_pgid && $3 == wanted_uid'
}

group_has_marker() {
    local wanted_pgid="$1" marker="$2"
    owned_group_rows "${wanted_pgid}" | grep -F -- "${marker}" >/dev/null
}

signal_owned_group() {
    local signal_name="$1" wanted_pgid="$2" process_id
    while read -r process_id; do
        [[ -z "${process_id}" ]] || kill "-${signal_name}" "${process_id}" 2>/dev/null || true
    done < <(owned_group_rows "${wanted_pgid}" | awk '{print $1}')
}

group_is_alive() {
    [[ -n "$(owned_group_rows "$1")" ]]
}

read -r delay_seconds target_time < <(
    python3 - <<'PY'
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

zone = ZoneInfo("America/Los_Angeles")
now = datetime.now(zone)
target = now.replace(hour=10, minute=0, second=0, microsecond=0)
if target <= now:
    target += timedelta(days=1)
print(max(0, int((target - now).total_seconds())), target.isoformat())
PY
)

# Refuse to arm if either caller accidentally supplied a group belonging to a
# different job.  Later, a missing/replaced group is treated as already done.
group_has_marker "${SAMPLE_PGID}" "${SAMPLE_MARKER}" || {
    echo "ERROR: sampler PGID ${SAMPLE_PGID} does not contain marker ${SAMPLE_MARKER}" >&2
    exit 1
}
group_has_marker "${TRAIN_PGID}" "${TRAIN_MARKER}" || {
    echo "ERROR: trainer PGID ${TRAIN_PGID} does not contain marker ${TRAIN_MARKER}" >&2
    exit 1
}

log "Armed for ${target_time} (${delay_seconds}s). Sampler PGID=${SAMPLE_PGID}; trainer PGID=${TRAIN_PGID}."
sleep "${delay_seconds}"

if group_has_marker "${SAMPLE_PGID}" "${SAMPLE_MARKER}"; then
    log "LA 10:00 reached; TERM only sampler UID=${UID}, PGID=${SAMPLE_PGID}."
    signal_owned_group TERM "${SAMPLE_PGID}"
    deadline=$((SECONDS + SAMPLE_TERM_GRACE_SECONDS))
    while group_is_alive "${SAMPLE_PGID}" && (( SECONDS < deadline )); do sleep 1; done
    if group_is_alive "${SAMPLE_PGID}"; then
        log "Sampler TERM grace expired; KILL only remaining UID=${UID}, PGID=${SAMPLE_PGID}."
        signal_owned_group KILL "${SAMPLE_PGID}"
    fi
else
    log "Sampler group already exited or changed; no sampler signal sent."
fi

if group_has_marker "${TRAIN_PGID}" "${TRAIN_MARKER}"; then
    printf 'requested_at=%s\nreason=scoped-la10-pause\npgid=%s\n' \
        "$(date -Is)" "${TRAIN_PGID}" > "${TRAIN_STOP_FILE}"
    log "Requested trainer exact-step checkpoint via ${TRAIN_STOP_FILE}."
    deadline=$((SECONDS + TRAIN_SAVE_GRACE_SECONDS))
    while group_is_alive "${TRAIN_PGID}" && (( SECONDS < deadline )); do sleep 1; done
    if group_is_alive "${TRAIN_PGID}"; then
        log "Trainer save grace expired; TERM only UID=${UID}, PGID=${TRAIN_PGID}."
        signal_owned_group TERM "${TRAIN_PGID}"
    else
        log "Trainer saved and exited cleanly."
    fi
else
    log "Trainer group already exited or changed; no stop request or signal sent."
fi

log "Scoped LA 10:00 pause finished."
