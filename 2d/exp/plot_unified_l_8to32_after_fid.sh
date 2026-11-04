#!/usr/bin/env bash
# Wait for the direct 8x8 -> 32x32 evaluation summary, then render the
# Unified-L stage-allocation comparison figures. This job is CPU-only.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
NEW_ROOT="${REPO_ROOT}/eval_results/unified_l_2stage_8to32_500500_convergence"
NEW_SUMMARY="${NEW_ROOT}/metrics_summary.tsv"
OUTPUT_DIR="${NEW_ROOT}/comparison_figures"
PLOT_SCRIPT="${REPO_ROOT}/paper/experiments_scripts/plot_unified_l_stage_convergence_comparison.py"
POLL_SECONDS="${POLL_SECONDS:-60}"

THREE_STAGE="${REPO_ROOT}/eval_results/unified_l_300300400_convergence/metrics_summary.tsv"
BOUNDARY="${REPO_ROOT}/eval_results/unified_l_boundary_convergence/metrics_summary.tsv"
TWO_STAGE="${REPO_ROOT}/eval_results/unified_l_2stage_500500_convergence/metrics_summary.tsv"

for required in "${PLOT_SCRIPT}" "${THREE_STAGE}" "${BOUNDARY}" "${TWO_STAGE}"; do
    [[ -s "${required}" ]] || { echo "ERROR: missing required input: ${required}" >&2; exit 1; }
done

echo "Waiting for completed direct 8->32 metrics: ${NEW_SUMMARY}"
while [[ ! -s "${NEW_SUMMARY}" ]]; do
    if ! tmux has-session -t eval_unified_l_8to32_500500_curve 2>/dev/null; then
        echo "ERROR: sampler/evaluator session exited without writing ${NEW_SUMMARY}" >&2
        exit 1
    fi
    sleep "${POLL_SECONDS}"
done

if [[ "$(wc -l < "${NEW_SUMMARY}")" -ne 9 ]]; then
    echo "ERROR: expected header + 8 checkpoints in ${NEW_SUMMARY}" >&2
    exit 1
fi

PYTHON="${REPO_ROOT}/.venv-rdd-eval/bin/python"
[[ -x "${PYTHON}" ]] || { echo "ERROR: evaluation Python not found: ${PYTHON}" >&2; exit 1; }

"${PYTHON}" "${PLOT_SCRIPT}" \
    --three-stage "${THREE_STAGE}" \
    --boundary "${BOUNDARY}" \
    --two-stage "${TWO_STAGE}" \
    --direct-two-stage "${NEW_SUMMARY}" \
    --output-dir "${OUTPUT_DIR}"

echo "Comparison figures complete: ${OUTPUT_DIR}"
