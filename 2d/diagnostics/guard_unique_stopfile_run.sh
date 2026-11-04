#!/usr/bin/env bash
# Supplementary hard-stop guard for one already-running RDD launch.
#
# Scope is defined by a unique absolute --stop-file path embedded in every
# torchrun worker's argv.  At the next America/Los_Angeles 07:00 it creates
# that file (the training loop then checkpoints and exits).  If any matching
# same-UID process remains after the grace periods, only the exact matching
# PIDs are TERM/KILLed.  It never matches by username alone, Python name, model
# name, GPU, or repository path.
set -Eeuo pipefail

: "${STOP_FILE:?set STOP_FILE to the unique absolute stop-file of this run}"
LA_TIMEZONE="America/Los_Angeles"
SAVE_GRACE_SECONDS="${SAVE_GRACE_SECONDS:-300}"
TERM_GRACE_SECONDS="${TERM_GRACE_SECONDS:-20}"
LOG_FILE="${LOG_FILE:-${STOP_FILE}.hard-guard.log}"

if [[ "${STOP_FILE}" != /* || "${STOP_FILE}" != *"/.la7-control/stop-"* ]]; then
  echo "ERROR: refusing unsafe STOP_FILE: ${STOP_FILE}" >&2
  exit 2
fi
mkdir -p "$(dirname "${STOP_FILE}")" "$(dirname "${LOG_FILE}")"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$1" | tee -a "${LOG_FILE}"
}

seconds_until_next_la_7am() {
  python3 - <<'PY'
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
zone = ZoneInfo("America/Los_Angeles")
now = datetime.now(zone)
target = now.replace(hour=7, minute=0, second=0, microsecond=0)
if target <= now:
    target += timedelta(days=1)
print(max(0, int((target - now).total_seconds())), target.isoformat())
PY
}

matching_pids() {
  # Args must contain the exact unique token. Exclude this guard and its
  # immediate parent even if a launcher happens to show the token in argv.
  ps -eo pid=,uid=,args= | awk \
    -v wanted_uid="${UID}" -v token="${STOP_FILE}" \
    -v self="$$" -v parent="${PPID}" \
    '$2 == wanted_uid && $1 != self && $1 != parent && index($0, token) { print $1 }'
}

signal_matches() {
  local signal_name="$1"
  local pids=()
  mapfile -t pids < <(matching_pids)
  for pid in "${pids[@]}"; do
    [[ "${pid}" =~ ^[0-9]+$ ]] || continue
    kill "-${signal_name}" "${pid}" 2>/dev/null || true
  done
}

run_is_alive() {
  [[ -n "$(matching_pids)" ]]
}

read -r delay target < <(seconds_until_next_la_7am)
log "Guarding exact stop token ${STOP_FILE}; next LA stop ${target} (${delay}s)."
sleep "${delay}"

printf 'requested_at=%s\ntimezone=%s\nmode=unique-stopfile-hard-guard\n' \
  "$(date -Is)" "${LA_TIMEZONE}" > "${STOP_FILE}"
log "LA 07:00 reached; stop-file created. Waiting up to ${SAVE_GRACE_SECONDS}s for checkpoint-and-exit."

deadline=$((SECONDS + SAVE_GRACE_SECONDS))
while run_is_alive && (( SECONDS < deadline )); do sleep 1; done
if ! run_is_alive; then
  log "No same-UID process carrying the unique stop token remains."
  exit 0
fi

log "Save grace expired; TERM only exact same-UID PIDs carrying the unique stop token."
signal_matches TERM
deadline=$((SECONDS + TERM_GRACE_SECONDS))
while run_is_alive && (( SECONDS < deadline )); do sleep 1; done
if run_is_alive; then
  log "TERM grace expired; KILL only exact same-UID PIDs carrying the unique stop token."
  signal_matches KILL
  sleep 1
  signal_matches KILL
fi

if run_is_alive; then
  log "ERROR: matching processes remain after KILL."
  exit 1
fi
log "Hard-stop guard complete."
