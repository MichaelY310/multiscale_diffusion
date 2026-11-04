#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${VIDEO_ENV_DIR:-${REPO_DIR}/.venv-rdd-video}/bin/activate"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
torchrun --standalone --nproc_per_node=2 "${SCRIPT_DIR}/test_rdd_distillation_ddp_gpu.py"
