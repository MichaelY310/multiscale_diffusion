#!/usr/bin/env bash
# Install the official Wan2.1 code and download the open student checkpoint,
# plus the optional 14B teacher used by the paper data recipe.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WAN_REPO_DIR="${WAN_REPO_DIR:-${REPO_DIR}/Wan2.1}"
PRETRAINED_DIR="${PRETRAINED_DIR:-${REPO_DIR}/pretrained_models/Wan2.1-T2V-1.3B}"
TEACHER_14B_DIR="${TEACHER_14B_DIR:-${REPO_DIR}/pretrained_models/Wan2.1-T2V-14B}"
DOWNLOAD_TEACHER_14B="${DOWNLOAD_TEACHER_14B:-0}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"

if [[ ! -d "${WAN_REPO_DIR}/.git" ]]; then
    git clone --depth 1 https://github.com/Wan-Video/Wan2.1.git "${WAN_REPO_DIR}"
else
    echo "Wan source already present: ${WAN_REPO_DIR}"
fi

if [[ "${INSTALL_DEPS}" == "1" ]]; then
    # flash-attn imports psutil while generating its package metadata, before
    # the rest of Wan's requirements can be installed.
    python -m pip install -U huggingface_hub ninja packaging psutil
    # flash-attn's build needs the already-installed torch in the active env.
    python -m pip install --no-build-isolation -r "${WAN_REPO_DIR}/requirements.txt"
fi

mkdir -p "${PRETRAINED_DIR}"
python - "${PRETRAINED_DIR}" <<'PY'
import sys
from huggingface_hub import snapshot_download

target = sys.argv[1]
snapshot_download(
    repo_id="Wan-AI/Wan2.1-T2V-1.3B",
    local_dir=target,
)
print(f"Wan checkpoint ready: {target}")
PY

for required in config.json Wan2.1_VAE.pth models_t5_umt5-xxl-enc-bf16.pth; do
    if [[ ! -e "${PRETRAINED_DIR}/${required}" ]]; then
        echo "ERROR: checkpoint is incomplete; missing ${required}" >&2
        exit 3
    fi
done
if ! find "${PRETRAINED_DIR}" -maxdepth 1 -type f -name 'diffusion_pytorch_model*' | grep -q .; then
    echo "ERROR: checkpoint is incomplete; missing diffusion model weights" >&2
    exit 3
fi

if [[ "${DOWNLOAD_TEACHER_14B}" == "1" ]]; then
    mkdir -p "${TEACHER_14B_DIR}"
    python - "${TEACHER_14B_DIR}" <<'PY'
import sys
from huggingface_hub import snapshot_download

target = sys.argv[1]
snapshot_download(
    repo_id="Wan-AI/Wan2.1-T2V-14B",
    local_dir=target,
)
print(f"Wan 14B teacher checkpoint ready: {target}")
PY
    for required in config.json Wan2.1_VAE.pth models_t5_umt5-xxl-enc-bf16.pth; do
        if [[ ! -e "${TEACHER_14B_DIR}/${required}" ]]; then
            echo "ERROR: 14B teacher checkpoint is incomplete; missing ${required}" >&2
            exit 3
        fi
    done
    if ! find "${TEACHER_14B_DIR}" -maxdepth 1 -type f -name 'diffusion_pytorch_model*' | grep -q .; then
        echo "ERROR: 14B teacher checkpoint is incomplete; missing diffusion weights" >&2
        exit 3
    fi
fi

git -C "${WAN_REPO_DIR}" rev-parse HEAD > "${REPO_DIR}/pretrained_models/WAN_SOURCE_REVISION.txt"

echo "Setup complete. Run the CPU contract tests:"
echo "  python ${SCRIPT_DIR}/test_rdd_video_cpu.py"
