#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../../.." && pwd -P)"
PYTHON_BIN="${UMDIFF_PYTHON:-$PROJECT_ROOT/.venv-fm-pf/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]] && ! command -v "$PYTHON_BIN" >/dev/null; then
  echo "Run setup_environment.sh first, or export UMDIFF_PYTHON=/path/to/python." >&2
  exit 1
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$PYTHON_BIN" "$EXPERIMENT_DIR/workflow.py" pf-train "$@"
