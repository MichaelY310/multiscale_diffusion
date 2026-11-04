#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SESSION="${RDD3D_TMUX_SESSION:-rdd3d_pilot}"
LOG="${ROOT}/logs/rdd3d_trellis2_pilot.log"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION}" >&2
    echo "Attach with: tmux attach -t ${SESSION}" >&2
    exit 2
fi

mkdir -p "${ROOT}/logs"
tmux new-session -d -s "${SESSION}" \
    "bash -lc 'cd \"${ROOT}\" && bash scripts/rdd3d/prepare_trellis2_pilot_data.sh 2>&1 | tee \"${LOG}\"'"

echo "Started tmux session: ${SESSION}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Log:    tail -f ${LOG}"
