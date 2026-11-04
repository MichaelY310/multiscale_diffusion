#!/bin/bash
set -euo pipefail
ROOT=/scr/yuheng/bugswarm/official_video_baselines_20261004
SNAPSHOT="$ROOT/reference/FastVideo-37dd31d5"
export PYTHONPATH="$ROOT:$SNAPSHOT:$ROOT/reference/Pyramid-Flow:/scr/yuheng/bugswarm/FastVideo/.venv/lib/python3.12/site-packages:$ROOT/official-env/lib/python3.12/site-packages"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec /scr/yuheng/bugswarm/FastVideo/.venv/bin/python "$@"
