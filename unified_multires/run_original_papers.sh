#!/usr/bin/env bash
# Default: inspect genuine author-CLI plans and their unresolved requirements.
# --entrypoint matched-core selects the executable pinned author core inside
# a shared experimental lifecycle. See PAIRED_EXPERIMENTS_ZH.md for limits.
set -euo pipefail
script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$script_root:$script_root/_deps${PYTHONPATH:+:$PYTHONPATH}"
experiment_python="${UMDIFF_PYTHON:-python3}"
exec "$experiment_python" "$script_root/tools/paired_experiments.py" --backend original "$@"
