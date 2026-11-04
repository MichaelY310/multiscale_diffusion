#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="/scr/yuheng/DevOpsBench/multiscale_diffusion"
LOG_FILE="${REPO_ROOT}/logs/archive_rdd_fm_unified_l_to_hf.log"

cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs"
set -o pipefail
"${REPO_ROOT}/.venv-rdd-eval/bin/python" -u \
    "${REPO_ROOT}/scripts/archive_rdd_fm_unified_l_to_hf.py" \
    2>&1 | tee -a "${LOG_FILE}"

