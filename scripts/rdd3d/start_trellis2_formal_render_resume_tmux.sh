#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SESSION="rdd3d_render_resume_2gpu"
LOG="${ROOT}/logs/${SESSION}.log"

mkdir -p "${ROOT}/logs"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION}" >&2
    exit 2
fi

tmux new-session -d -s "${SESSION}" \
    "bash -lc 'cd \"${ROOT}\" && bash scripts/rdd3d/resume_trellis2_formal_render_2gpu.sh 2>&1 | tee \"${LOG}\"'"

echo "Started ${SESSION}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Log:    tail -f ${LOG}"
