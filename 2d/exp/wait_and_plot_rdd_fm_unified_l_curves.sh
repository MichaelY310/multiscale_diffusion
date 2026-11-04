#!/usr/bin/env bash
# Plot the combined curve after both eight-checkpoint evaluations complete.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
POLL_SECONDS="${POLL_SECONDS:-120}"
SUMMARY_A="${REPO_ROOT}/eval_results/rdd_fm_unified_l_corrected_300300400_convergence/metrics_summary.tsv"
SUMMARY_B="${REPO_ROOT}/eval_results/rdd_fm_unified_l_corrected_400300300_convergence/metrics_summary.tsv"
OUTPUT_PREFIX="${REPO_ROOT}/eval_results/rdd_fm_unified_l_corrected_allocation_comparison/rdd_fm_unified_l_fid_curves"
PLOT_SCRIPT="${REPO_ROOT}/2d/plot_rdd_fm_unified_l_allocation_curves.py"
LOG_FILE="${REPO_ROOT}/logs/plot_rdd_fm_unified_l_curves.log"

mkdir -p "$(dirname "${OUTPUT_PREFIX}")" "${REPO_ROOT}/logs"

summary_complete() {
    local path="$1"
    [[ -s "${path}" ]] && [[ "$(wc -l < "${path}")" -eq 9 ]]
}

echo "[$(date -Is)] waiting for both eight-point summaries" | tee -a "${LOG_FILE}"
until summary_complete "${SUMMARY_A}" && summary_complete "${SUMMARY_B}"; do
    sleep "${POLL_SECONDS}"
done

python "${PLOT_SCRIPT}" \
    --summary-300300400 "${SUMMARY_A}" \
    --summary-400300300 "${SUMMARY_B}" \
    --output-prefix "${OUTPUT_PREFIX}" 2>&1 | tee -a "${LOG_FILE}"

echo "[$(date -Is)] combined curves ready: ${OUTPUT_PREFIX}.png" | tee -a "${LOG_FILE}"

