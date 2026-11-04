#!/usr/bin/env bash
# One-time bootstrap for RDD-Multistage-100-300-600 evaluation.
#
# Everything is kept under the repository root:
#   .venv-rdd-eval/                         Python environment
#   .cache/huggingface/                     Hugging Face cache
#   checkpoints/RDD-Multistage-100-300-600/ three RDD checkpoints
#   checkpoints/pretrained/                 SD VAE
#   eval_results/reference/                 ADM reference NPZ + Inception graph

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv-rdd-eval}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"
HF_REPO_ID="${HF_REPO_ID:-yang335/search_agent_hl_to_i18n_transport}"

export HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${ROOT_DIR}/.cache/pip}"

CHECKPOINT_FILES=(
  "checkpoints/RDD-Multistage-100-300-600/checkpoints_32x32/0400000.pt"
  "checkpoints/RDD-Multistage-100-300-600/checkpoints_16x16/0400000.pt"
  "checkpoints/RDD-Multistage-100-300-600/checkpoints_8x8/0400000.pt"
)

VAE_DIR="${ROOT_DIR}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema"
REFERENCE_DIR="${ROOT_DIR}/eval_results/reference"
REFERENCE_FILE="${REFERENCE_DIR}/VIRTUAL_imagenet256_labeled.npz"
INCEPTION_FILE="${REFERENCE_DIR}/classify_image_graph_def.pb"
REFERENCE_URL="https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz"
INCEPTION_URL="https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb"

echo "============================================================"
echo "RDD evaluation bootstrap"
echo "Repository root: ${ROOT_DIR}"
echo "Virtualenv: ${VENV_DIR}"
echo "PyTorch wheels: ${PYTORCH_INDEX_URL}"
echo "============================================================"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} was not found. Install Python 3.9-3.12 first." >&2
  exit 1
fi

"${PYTHON_BIN}" -c 'import sys; assert (3, 9) <= sys.version_info[:2] <= (3, 12), f"Python 3.9-3.12 required, got {sys.version.split()[0]}"'

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo
  echo "Creating virtual environment"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

echo
echo "Installing Python dependencies"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  --index-url "${PYTORCH_INDEX_URL}"
python -m pip install --upgrade \
  'numpy<2' \
  'pillow>=9.5' \
  'diffusers>=0.25' \
  'transformers>=4.35' \
  'accelerate>=0.24' \
  'safetensors>=0.4' \
  'timm>=0.9' \
  'tqdm>=4.65' \
  'huggingface_hub>=0.20' \
  scipy requests
python -m pip install --upgrade 'tensorflow-cpu>=2.15,<2.18'

echo
echo "Checking CUDA and evaluator imports"
python - <<'PY'
import torch
import tensorflow as tf
import diffusers
import timm

if not torch.cuda.is_available():
    raise SystemExit(
        "PyTorch cannot see CUDA. If this server needs a different wheel, rerun with "
        "PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128."
    )
print(f"torch={torch.__version__}, CUDA runtime={torch.version.cuda}, GPUs={torch.cuda.device_count()}")
print(f"compiled CUDA architectures={torch.cuda.get_arch_list()}")
cuda_probe = torch.arange(8, device="cuda").square().sum()
torch.cuda.synchronize()
print(f"CUDA kernel probe={cuda_probe.item()}")
print(f"tensorflow={tf.__version__}, diffusers={diffusers.__version__}, timm={timm.__version__}")
if torch.cuda.device_count() < 2:
    raise SystemExit(f"Two GPUs are required for the requested sampler, found {torch.cuda.device_count()}.")
PY

if [[ -z "${HF_TOKEN:-}" && -t 0 ]]; then
  echo
  read -r -s -p "Hugging Face token (press Enter if the repo is public): " entered_hf_token || true
  echo
  if [[ -n "${entered_hf_token:-}" ]]; then
    export HF_TOKEN="${entered_hf_token}"
  fi
fi

echo
echo "Downloading three RDD checkpoints from ${HF_REPO_ID}"
for remote_file in "${CHECKPOINT_FILES[@]}"; do
  python "${SCRIPT_DIR}/hf_download.py" \
    --repo "${HF_REPO_ID}" \
    --file "${remote_file}" \
    --out "${ROOT_DIR}"

  local_file="${ROOT_DIR}/${remote_file}"
  if [[ ! -s "${local_file}" ]]; then
    echo "ERROR: expected checkpoint was not created: ${local_file}" >&2
    exit 1
  fi
done

echo
echo "Downloading SD VAE to ${VAE_DIR}"
python - "${VAE_DIR}" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="stabilityai/sd-vae-ft-ema",
    local_dir=sys.argv[1],
    token=os.environ.get("HF_TOKEN"),
)
PY
if [[ ! -s "${VAE_DIR}/config.json" ]]; then
  echo "ERROR: VAE download is incomplete: ${VAE_DIR}" >&2
  exit 1
fi

download_large_file() {
  local url="$1"
  local destination="$2"
  local min_bytes="$3"
  local partial="${destination}.part"
  local current_size=0

  mkdir -p "$(dirname "${destination}")"
  if [[ -f "${destination}" ]]; then
    current_size="$(stat -c '%s' "${destination}")"
  fi
  if [[ "${current_size}" -ge "${min_bytes}" ]]; then
    echo "Already present: ${destination} (${current_size} bytes)"
    return
  fi
  if [[ -f "${destination}" && ! -f "${partial}" ]]; then
    mv "${destination}" "${partial}"
  fi

  echo "Downloading ${url}"
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 5 --continue-at - --output "${partial}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget --continue --output-document="${partial}" "${url}"
  else
    echo "ERROR: curl or wget is required for the ADM files." >&2
    exit 1
  fi

  current_size="$(stat -c '%s' "${partial}")"
  if [[ "${current_size}" -lt "${min_bytes}" ]]; then
    echo "ERROR: downloaded file is unexpectedly small: ${partial} (${current_size} bytes)" >&2
    exit 1
  fi
  mv "${partial}" "${destination}"
}

echo
echo "Downloading ADM ImageNet-256 reference data and Inception graph"
download_large_file "${REFERENCE_URL}" "${REFERENCE_FILE}" 1900000000
download_large_file "${INCEPTION_URL}" "${INCEPTION_FILE}" 50000000

echo
echo "Verifying downloaded artifacts"
for remote_file in "${CHECKPOINT_FILES[@]}"; do
  ls -lh "${ROOT_DIR}/${remote_file}"
done
ls -lh "${VAE_DIR}/config.json" "${REFERENCE_FILE}" "${INCEPTION_FILE}"

echo
echo "Bootstrap complete. Run:"
echo "  bash ${ROOT_DIR}/2d/exp/eval_ms100300600_400k_2gpu.sh"
