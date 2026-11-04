#!/usr/bin/env bash
# Install the official VBench implementation in an isolated virtualenv and
# materialize the exact standard-suite generation manifest used by this repo.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VBENCH_DIR="${VBENCH_DIR:-${REPO_DIR}/third_party/VBench}"
VBENCH_ENV="${VBENCH_ENV:-${REPO_DIR}/third_party/vbench_env}"
VBENCH_REVISION="${VBENCH_REVISION:-master}"
MANIFEST="${MANIFEST:-${REPO_DIR}/rdd_video_data/vbench/vbench_manifest.jsonl}"
INSTALL_DETECTRON2="${INSTALL_DETECTRON2:-0}"

mkdir -p "$(dirname "${VBENCH_DIR}")"
if [[ ! -d "${VBENCH_DIR}/.git" ]]; then
  git clone https://github.com/Vchitect/VBench.git "${VBENCH_DIR}"
fi
git -C "${VBENCH_DIR}" fetch --tags origin
git -C "${VBENCH_DIR}" checkout "${VBENCH_REVISION}"

# Do not modify the Wan training environment.  Reuse its CUDA-enabled torch
# through system-site-packages and isolate the remaining VBench dependencies.
if [[ ! -x "${VBENCH_ENV}/bin/python" ]]; then
  python -m venv --system-site-packages "${VBENCH_ENV}"
fi
"${VBENCH_ENV}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VBENCH_ENV}/bin/python" -m pip install -e "${VBENCH_DIR}"

if [[ "${INSTALL_DETECTRON2}" == "1" ]]; then
  "${VBENCH_ENV}/bin/python" -m pip install \
    'detectron2@git+https://github.com/facebookresearch/detectron2.git'
else
  echo "NOTE: detectron2 was not installed. It is required by some semantic dimensions."
  echo "      The official package supports CUDA 11.x/12.1; use a compatible eval node and rerun:"
  echo "      INSTALL_DETECTRON2=1 bash rdd_video/setup_video_eval.sh"
fi

"${VBENCH_ENV}/bin/python" "${SCRIPT_DIR}/build_vbench_manifest.py" \
  --vbench-dir "${VBENCH_DIR}" \
  --output "${MANIFEST}" \
  --base-seed 2026 \
  --samples-per-prompt 5 \
  --temporal-flickering-samples 25

revision="$(git -C "${VBENCH_DIR}" rev-parse HEAD)"
printf '%s\n' "${revision}" > "${MANIFEST}.vbench_revision"
"${VBENCH_ENV}/bin/python" - <<'PY'
import torch
import vbench
print(f"VBench import OK; torch={torch.__version__}; cuda={torch.version.cuda}")
PY

echo "VBench ready: ${VBENCH_DIR} @ ${revision}"
echo "Environment: ${VBENCH_ENV}"
echo "Manifest: ${MANIFEST}"

