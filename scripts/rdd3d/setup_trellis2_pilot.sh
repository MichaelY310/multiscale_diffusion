#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
THIRD_PARTY="${ROOT}/third_party"
VENV="${ROOT}/.venv_rdd3d"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
TRELLIS_COMMIT="${TRELLIS_COMMIT:-75fbf0183001ed9876c8dbb35de6b68552ee08bd}"
FLEXGEMM_COMMIT="${FLEXGEMM_COMMIT:-6dd94a859c26ee8246888502eada3dd8ad85532e}"
CUMESH_COMMIT="${CUMESH_COMMIT:-12289e1062f0603f2f0d0771b02e1395d247f26f}"

mkdir -p "${THIRD_PARTY}" "${ROOT}/logs"

clone_at() {
    local url="$1"
    local path="$2"
    local commit="$3"
    if [[ ! -d "${path}/.git" ]]; then
        git clone --recursive "${url}" "${path}"
    fi
    git -C "${path}" fetch --all --tags
    git -C "${path}" checkout "${commit}"
    git -C "${path}" submodule update --init --recursive
}

clone_at https://github.com/microsoft/TRELLIS.2.git "${THIRD_PARTY}/TRELLIS.2" "${TRELLIS_COMMIT}"
clone_at https://github.com/JeffreyXiang/FlexGEMM.git "${THIRD_PARTY}/FlexGEMM" "${FLEXGEMM_COMMIT}"
clone_at https://github.com/JeffreyXiang/CuMesh.git "${THIRD_PARTY}/CuMesh" "${CUMESH_COMMIT}"
if [[ ! -d "${THIRD_PARTY}/nvdiffrast/.git" ]]; then
    git clone --branch v0.4.0 --recursive https://github.com/NVlabs/nvdiffrast.git "${THIRD_PARTY}/nvdiffrast"
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
    python3 -m venv "${VENV}"
fi
PYTHON="${VENV}/bin/python"
PIP="${VENV}/bin/pip"

"${PIP}" install --upgrade pip setuptools wheel
if ! "${PYTHON}" -c 'import torch, torchvision; assert torch.__version__.startswith("2.10.0+cu130"); assert torchvision.__version__.startswith("0.25.0+cu130")' 2>/dev/null; then
    "${PIP}" install --upgrade --force-reinstall torch==2.10.0 torchvision==0.25.0 \
        --index-url "${PYTORCH_INDEX_URL}"
fi
"${PIP}" install \
    pillow plyfile imageio imageio-ffmpeg tqdm easydict opencv-python-headless \
    pandas open3d objaverse huggingface_hub open_clip_torch trimesh ninja \
    zstandard transformers tensorboard lpips kornia timm
# Keep the PyTorch/CUDA build selected above. Installing without dependencies
# prevents pip from silently replacing torch with an incompatible wheel.
"${PIP}" install --no-deps xformers==0.0.35 --index-url "${PYTORCH_INDEX_URL}"
"${PIP}" install \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

export CUDA_HOME
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
"${PIP}" install "${THIRD_PARTY}/FlexGEMM" --no-build-isolation
"${PIP}" install "${THIRD_PARTY}/nvdiffrast" --no-build-isolation
"${PIP}" install "${THIRD_PARTY}/CuMesh" --no-build-isolation
"${PIP}" install "${THIRD_PARTY}/TRELLIS.2/o-voxel" --no-build-isolation

cd "${THIRD_PARTY}/TRELLIS.2"
SPARSE_CONV_BACKEND=flex_gemm PYTHONPATH=. "${PYTHON}" - <<'PY'
import torch
import o_voxel
import flex_gemm
import cumesh
import nvdiffrast.torch
import trellis2.models

print("TRELLIS.2 pilot environment OK")
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable")
PY

echo "Environment: ${VENV}"
echo "TRELLIS.2:   ${THIRD_PARTY}/TRELLIS.2 @ ${TRELLIS_COMMIT}"
