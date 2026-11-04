#!/usr/bin/env bash
# Download the legacy RDD-FM Unified XL/2 v2 400K checkpoint into this repo.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
HF_REPO_ID="${HF_REPO_ID:-yang335/search_agent_hl_to_i18n_transport}"
REMOTE_FILE="checkpoints/RDD-FM-Unified-XL-v2/0400000.pt"
LOCAL_FILE="${ROOT_DIR}/${REMOTE_FILE}"

if [[ -f "${ROOT_DIR}/.venv-rdd-eval/bin/activate" ]]; then
  source "${ROOT_DIR}/.venv-rdd-eval/bin/activate"
fi

if ! python -c 'import huggingface_hub' >/dev/null 2>&1; then
  echo "ERROR: huggingface_hub is missing." >&2
  echo "Run scripts/download_rdd_multistage_100_300_600_400k.sh once, or install huggingface_hub." >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
mkdir -p "$(dirname "${LOCAL_FILE}")"

echo "Downloading ${HF_REPO_ID}/${REMOTE_FILE}"
echo "Expected size: about 10.8 GB (the training checkpoint includes optimizer state)."
python "${SCRIPT_DIR}/hf_download.py" \
  --repo "${HF_REPO_ID}" \
  --file "${REMOTE_FILE}" \
  --out "${ROOT_DIR}"

size="$(stat -c '%s' "${LOCAL_FILE}")"
if [[ "${size}" -lt 10000000000 ]]; then
  echo "ERROR: checkpoint is unexpectedly small: ${LOCAL_FILE} (${size} bytes)" >&2
  exit 1
fi

ls -lh "${LOCAL_FILE}"
echo "Checkpoint ready: ${LOCAL_FILE}"

