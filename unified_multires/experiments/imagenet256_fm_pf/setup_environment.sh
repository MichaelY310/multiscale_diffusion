#!/usr/bin/env bash
set -euo pipefail
export PIP_NO_CACHE_DIR=1
EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../../.." && pwd -P)"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}"
ENV_DIR="${FM_PF_ENV_DIR:-$PROJECT_ROOT/.venv-fm-pf}"
"$BOOTSTRAP_PYTHON" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python3.10-3.12 (recommended3.11 or3.12)"'
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$BOOTSTRAP_PYTHON" -m venv "$ENV_DIR"
fi
"$ENV_DIR/bin/python" -m pip install --upgrade pip 'setuptools>=68' wheel
"$ENV_DIR/bin/python" -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
"$ENV_DIR/bin/python" -m pip install -r "$EXPERIMENT_DIR/requirements.txt"
"$ENV_DIR/bin/python" -m pip install --no-deps --no-build-isolation -e "$PROJECT_ROOT/unified_multires"
"$ENV_DIR/bin/python" -m pip check
"$ENV_DIR/bin/python" -c 'import torch,torchvision,timm; from diffusers import AutoencoderKL; from umdiff.cli import make_method; print("Imports OK; torch", torch.__version__)'
echo "Environment ready: $ENV_DIR"
echo "Select an available authorized GPU via CUDA_VISIBLE_DEVICES before launching experiments."
