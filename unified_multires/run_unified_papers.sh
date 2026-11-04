#!/usr/bin/env bash
# Same manifest as the original launcher. Default: inspect unified-CLI plans.
# --entrypoint matched-core selects Method components with the same controlled
# reader/lifecycle as the original source-core launcher.
set -euo pipefail
script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$script_root:$script_root/_deps${PYTHONPATH:+:$PYTHONPATH}"
experiment_python="${UMDIFF_PYTHON:-python3}"
exec "$experiment_python" "$script_root/tools/paired_experiments.py" --backend unified "$@"
