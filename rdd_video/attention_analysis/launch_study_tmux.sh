#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SESSION="${SESSION:-rdd_wan_harp_probe}"
LOG="${LOG:-${REPO_DIR}/eval_results/rdd_wan_attention_routing/launcher.log}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${SESSION}" >&2
    exit 2
fi
mkdir -p "$(dirname "${LOG}")"
payload="cd $(printf '%q' "${REPO_DIR}") && bash $(printf '%q' "${SCRIPT_DIR}/run_study_2gpu.sh") 2>&1 | tee $(printf '%q' "${LOG}")"
tmux new-session -d -s "${SESSION}" "bash -lc $(printf '%q' "${payload}")"
echo "Started tmux session: ${SESSION}"
echo "Watch: tmux attach -t ${SESSION}"
echo "Log: ${LOG}"

