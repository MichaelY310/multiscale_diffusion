#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SESSION="${RDD3D_INTERFACE_ENDPOINT_SESSION:-rdd3d_stage_interface_endpoint_head}"
LOG="${ROOT}/logs/rdd3d_stage_interface_endpoint_head.log"

mkdir -p "${ROOT}/logs"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION}" >&2
    exit 2
fi
tmux new-session -d -s "${SESSION}" \
    "cd '${ROOT}' && bash scripts/rdd3d/run_dynamic_sparse_stage_interface_endpoint_head_2gpu.sh 2>&1 | tee '${LOG}'"
echo "Started ${SESSION}"
echo "Log: ${LOG}"
