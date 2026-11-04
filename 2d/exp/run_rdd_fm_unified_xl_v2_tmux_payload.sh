#!/usr/bin/env bash
# Non-interactive payload intended to be launched inside the fm_eval tmux session.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_FILE="${SCRIPT_DIR}/eval_rdd_fm_unified_xl_v2_400k_2gpu.log"

cd "${ROOT_DIR}"
{
  bash scripts/download_rdd_fm_unified_xl_v2_400k.sh
  bash 2d/exp/eval_rdd_fm_unified_xl_v2_400k_2gpu.sh
} 2>&1 | tee "${LOG_FILE}"

