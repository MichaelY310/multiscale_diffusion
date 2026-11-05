#!/usr/bin/env bash
# CPU-only gate: start the 400/300/300 evaluation on GPUs 0-3 only after the
# current direct 8->32 evaluation (including all metrics) is complete.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
UPSTREAM_SUMMARY="${REPO_ROOT}/eval_results/unified_l_2stage_8to32_500500_convergence/metrics_summary.tsv"
POLL_SECONDS="${POLL_SECONDS:-60}"

echo "Waiting for header + 8 rows in ${UPSTREAM_SUMMARY}"
while true; do
    if [[ -s "${UPSTREAM_SUMMARY}" ]] && [[ "$(wc -l < "${UPSTREAM_SUMMARY}")" -eq 9 ]]; then
        break
    fi
    if ! tmux has-session -t eval_unified_l_8to32_500500_curve 2>/dev/null; then
        echo "ERROR: upstream evaluation exited before producing a complete summary" >&2
        exit 1
    fi
    sleep "${POLL_SECONDS}"
done

echo "Upstream complete; snapshotting available 400/300/300 checkpoints now."
export GPU_IDS="0,1,2,3"
export WORLD_SIZE="4"
exec bash "${REPO_ROOT}/2d/exp/eval_unified_l_400300300_available_4gpu.sh"
