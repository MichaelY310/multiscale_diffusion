#!/usr/bin/env bash
# CPU-only preparation/audit. This entrypoint has no GPU execution action.
set -euo pipefail
script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
experiment_python="${UMDIFF_PYTHON:-python3}"
exec "$experiment_python" "$script_root/tools/prepare_gpu_checks.py" "$@"
