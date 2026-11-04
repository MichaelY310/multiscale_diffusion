#!/usr/bin/env bash
# Create an isolated Blackwell-capable video environment, then install Wan2.1
# and download the requested checkpoints. This does not touch the 2D training
# or FM evaluation environments.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VIDEO_ENV_DIR="${VIDEO_ENV_DIR:-${REPO_DIR}/.venv-rdd-video}"
TORCH_VERSION="${TORCH_VERSION:-2.13.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.28.0}"

if [[ ! -x "${VIDEO_ENV_DIR}/bin/python" ]]; then
  python3 -m venv "${VIDEO_ENV_DIR}"
fi
source "${VIDEO_ENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

# PyPI's Linux PyTorch 2.13 wheel is CUDA 13.0, which is required for the RTX
# PRO 6000 Blackwell host used here. Override the versions on another host.
python -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}"

DOWNLOAD_TEACHER_14B="${DOWNLOAD_TEACHER_14B:-0}" \
INSTALL_DEPS=1 \
bash "${SCRIPT_DIR}/setup_wan21.sh"

python - <<'PY'
import torch
import torchvision
import flash_attn
print(
    f"RDD video environment ready: torch={torch.__version__}, "
    f"cuda={torch.version.cuda}, torchvision={torchvision.__version__}, "
    f"flash_attn={getattr(flash_attn, '__version__', 'import-ok')}"
)
PY

PYTHONIOENCODING=utf-8 RUN_GPU_SMOKE=0 bash "${SCRIPT_DIR}/run_rdd_wan_smoke.sh"
echo "Activate later with: source ${VIDEO_ENV_DIR}/bin/activate"
