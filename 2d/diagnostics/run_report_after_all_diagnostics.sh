#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv-rdd-eval/bin/python}"
cd "${REPO_ROOT}"
mkdir -p logs

sessions=(
  rdd_stage_cpu_diag
  predx0_300_vs_100_after_gpu0_idle
  teacher_forced_after_predx0
  local_solver_after_teacher
  class_consistency_after_local_solver
  vae_frequency_control_after_gpu1_idle
)
for session in "${sessions[@]}"; do
  while tmux has-session -t "${session}" 2>/dev/null; do
    echo "[$(date -Is)] waiting for ${session}"
    sleep 60
  done
done

exec "${PYTHON}" -u 2d/diagnostics/build_stage_allocation_report.py \
  --repo-root "${REPO_ROOT}" \
  --bundle-dir "${REPO_ROOT}/eval_results/diagnostics/RDD_STAGE_ALLOCATION_REPORT_BUNDLE"
