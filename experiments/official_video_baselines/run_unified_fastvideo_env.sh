#!/bin/bash
set -euo pipefail
ROOT=/scr/yuheng/bugswarm/official_video_baselines_20261004
HELPERS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MIGRATED=$(cd -- "$HELPERS/../../.." && pwd)
export PYTHONPATH="$HELPERS:$ROOT:$ROOT/reference/FastVideo-37dd31d5:$ROOT/reference/Pyramid-Flow:$MIGRATED/FastVideo-RDD-minimal:/scr/yuheng/bugswarm/FastVideo/.venv/lib/python3.12/site-packages:$ROOT/official-env/lib/python3.12/site-packages"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec /scr/yuheng/bugswarm/FastVideo/.venv/bin/python "$@"
