#!/usr/bin/env bash

set -Eeuo pipefail
set -o pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/logs/rdd_unified_l_300300400_rdd_dmd_full_v2_gb256_4567.log}"

cd "${REPO_ROOT}"
mkdir -p "$(dirname "${LOG_FILE}")"
bash 2d/exp/prepare_and_train_rdd_dmd_full_v2_4567.sh 2>&1 | tee -a "${LOG_FILE}"
