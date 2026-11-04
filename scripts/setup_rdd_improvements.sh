#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-rdd-train/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: no Python interpreter found. Set PYTHON_BIN explicitly." >&2
  exit 2
fi
PRETRAINED="$ROOT/checkpoints/pretrained"
DINOV2_REPO="$PRETRAINED/dinov2"
DINOV2_WEIGHTS="$PRETRAINED/dinov2_vitb14_pretrain.pth"

mkdir -p "$PRETRAINED"

if [[ ! -d "$DINOV2_REPO/.git" ]]; then
  git clone --depth 1 https://github.com/facebookresearch/dinov2.git "$DINOV2_REPO"
fi

if [[ ! -s "$DINOV2_WEIGHTS" ]]; then
  curl --fail --location --retry 5 --continue-at - \
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth \
    --output "$DINOV2_WEIGHTS"
fi

"$PYTHON_BIN" - <<'PY'
import importlib
required = ["torch", "torchvision", "timm", "diffusers", "PIL", "numpy"]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    raise SystemExit("Missing/incompatible Python dependencies:\n  " + "\n  ".join(missing))
print("RDD improvement dependencies import successfully.")
PY

echo "DINOv2 repo:    $DINOV2_REPO"
echo "DINOv2 weights: $DINOV2_WEIGHTS"
echo "Python:         $PYTHON_BIN"
