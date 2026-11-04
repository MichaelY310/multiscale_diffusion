#!/bin/bash
set -euo pipefail
ROOT=/scr/yuheng/bugswarm/official_video_baselines_20261004
export PYTHONPATH="$ROOT/official-env/lib/python3.12/site-packages:/scr/yuheng/bugswarm/FastVideo/.venv/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "$ROOT/official-env/bin/python" "$@"
