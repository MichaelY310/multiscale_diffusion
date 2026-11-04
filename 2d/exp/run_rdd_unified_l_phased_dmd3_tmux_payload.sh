#!/usr/bin/env bash
# Server-side tmux payload.  Keeping the pipeline here avoids SSH/PowerShell
# quoting changing which process owns the training job.

set -Eeuo pipefail
set -o pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/logs/rdd_unified_l_phased_dmd3_4567.log}"

cd "${REPO_ROOT}"
mkdir -p "$(dirname "${LOG_FILE}")"
bash 2d/exp/train_rdd_unified_l_phased_dmd3_4567.sh 2>&1 | tee -a "${LOG_FILE}"

