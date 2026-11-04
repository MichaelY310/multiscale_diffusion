#!/usr/bin/env bash
# Sample a merged 2/2/1 RDD-Wan student with exactly five conditional NFEs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VIDEO_ENV_DIR="${VIDEO_ENV_DIR:-${REPO_DIR}/.venv-rdd-video}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to the merged distilled RDD-Wan checkpoint}"
PRETRAINED_DIR="${PRETRAINED_DIR:-${REPO_DIR}/pretrained_models/Wan2.1-T2V-1.3B}"

source "${VIDEO_ENV_DIR}/bin/activate"
export PYTHONPATH="${REPO_DIR}/Wan2.1:${REPO_DIR}/rdd_video:${PYTHONPATH:-}"
python "${SCRIPT_DIR}/sample_rdd_wan.py" \
  --checkpoint "${CHECKPOINT}" \
  --pretrained-dir "${PRETRAINED_DIR}" \
  --prompt "${PROMPT:-A cinematic tracking shot of a corgi running through a field of flowers}" \
  --steps-per-stage 2,2,1 \
  --guidance-scale 1.0 \
  --solver euler \
  --distilled-boundary-clean \
  --device "${DEVICE:-0}" \
  --seed "${SEED:-42}" \
  --output "${OUTPUT:-${REPO_DIR}/output_rdd_video/distilled_221.mp4}"
