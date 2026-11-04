#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SESSION="${SESSION:-rdd_2d_cross_stage_probe}"
LOG="${REPO_DIR}/eval_results/rdd_2d_cross_stage/logs/launcher.log"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${SESSION}" >&2
    exit 2
fi
mkdir -p "$(dirname "${LOG}")"
payload="cd $(printf '%q' "${REPO_DIR}") && bash $(printf '%q' "${SCRIPT_DIR}/run_rdd_2d_cross_stage_probe.sh") 2>&1 | tee $(printf '%q' "${LOG}")"
tmux new-session -d -s "${SESSION}" "bash -lc $(printf '%q' "${payload}")"
echo "Started: ${SESSION}"
echo "Watch: tmux attach -t ${SESSION}"
echo "Log: ${LOG}"

