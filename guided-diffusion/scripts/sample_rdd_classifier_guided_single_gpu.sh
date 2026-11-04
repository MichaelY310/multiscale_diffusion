#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

python scripts/classifier_sample_with_drift_imagenet256.py   --ckpt_64 "$CKPT_64"   --ckpt_128 "$CKPT_128"   --ckpt_256 "$CKPT_256"   --classifier_64 "$CLASSIFIER_64"   --classifier_128 "$CLASSIFIER_128"   --classifier_256 "$CLASSIFIER_256"   --class_label "${CLASS_LABEL:-207}"   --num_samples "${NUM_SAMPLES:-4}"   --output "${OUTPUT:-guided_diffusion/samples_rdd_classifier_guided.png}"
