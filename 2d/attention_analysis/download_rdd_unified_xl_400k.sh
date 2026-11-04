#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.venv-rdd-attn/bin/python}"
TARGET="${REPO_DIR}/checkpoints/RDD-Unified-XL/0400000.pt"

mkdir -p "$(dirname "${TARGET}")"
if [[ -s "${TARGET}" ]]; then
    echo "Already present: ${TARGET}"
    exit 0
fi

"${PYTHON_BIN}" - "${REPO_DIR}" <<'PY'
import sys
from huggingface_hub import hf_hub_download

print(
    hf_hub_download(
        repo_id="yang335/search_agent_hl_to_i18n_transport",
        filename="checkpoints/RDD-Unified-XL/0400000.pt",
        local_dir=sys.argv[1],
    )
)
PY
test -s "${TARGET}"
echo "Download complete: ${TARGET}"

