#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
REPO_ID="yang335/search_agent_hl_to_i18n_transport"
FILENAME="checkpoints/DiT-XL-2/0400000.pt"

cd "$ROOT"
mkdir -p checkpoints/DiT-XL-2 eval_results/logs

if [[ -s "$FILENAME" ]]; then
  echo "Checkpoint already exists: $ROOT/$FILENAME"
  exit 0
fi

exec "$ROOT/.venv-rdd-video/bin/hf" download \
  "$REPO_ID" \
  "$FILENAME" \
  --local-dir "$ROOT"
