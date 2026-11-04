#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SESSION="${RDD3D_FINAL_TEST_SESSION:-rdd3d_final_test_prepare}"
LOG="${ROOT}/logs/rdd3d_final_test_prepare.log"

mkdir -p "${ROOT}/logs"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION}" >&2
    exit 2
fi
tmux new-session -d -s "${SESSION}" \
    "cd '${ROOT}' && bash scripts/rdd3d/prepare_dynamic_sparse_final_test.sh 2>&1 | tee '${LOG}'"
echo "Started ${SESSION}"
echo "Log: ${LOG}"
