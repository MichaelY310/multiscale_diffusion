#!/usr/bin/env bash
# Minimal, isolated environment for the RDD-Wan attention-routing probe.
# Downloads only the 5.7 GB Wan transformer; T5 and VAE are not required.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_DIR="${RDD_ATTN_ENV_DIR:-${REPO_DIR}/.venv-rdd-attn}"
WAN_REPO_DIR="${WAN_REPO_DIR:-${REPO_DIR}/Wan2.1}"
MODEL_DIR="${WAN_MODEL_DIR:-${REPO_DIR}/pretrained_models/Wan2.1-T2V-1.3B-transformer-only}"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    python3 -m venv --system-site-packages "${ENV_DIR}"
fi
source "${ENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install \
    'transformers>=4.49.0' 'accelerate>=1.1.1' \
    easydict einops ftfy imageio imageio-ffmpeg opencv-python-headless tqdm

if [[ ! -d "${WAN_REPO_DIR}/.git" ]]; then
    git clone --depth 1 https://github.com/Wan-Video/Wan2.1.git "${WAN_REPO_DIR}"
fi

mkdir -p "${MODEL_DIR}"
python - "${MODEL_DIR}" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Wan-AI/Wan2.1-T2V-1.3B",
    local_dir=sys.argv[1],
    allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"],
)
PY

test -s "${MODEL_DIR}/config.json"
test -s "${MODEL_DIR}/diffusion_pytorch_model.safetensors"
python -m py_compile \
    "${SCRIPT_DIR}/routing_metrics.py" \
    "${SCRIPT_DIR}/probe_rdd_wan_routing.py" \
    "${SCRIPT_DIR}/benchmark_harp_flex.py"
python - <<PY
import torch
print("probe environment ready", torch.__version__, torch.version.cuda, torch.cuda.is_available())
print("Wan source: ${WAN_REPO_DIR}")
print("Wan transformer: ${MODEL_DIR}")
PY
