# Source this file from Bash; use the chosen Python without installing packages.
UMDIFF_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -f "$UMDIFF_ROOT/.env.local.sh" ]]; then
  source "$UMDIFF_ROOT/.env.local.sh"
fi
export UMDIFF_ROOT
export UMDIFF_PYTHON="${UMDIFF_PYTHON:-python3}"
export PYTHONPATH="$UMDIFF_ROOT:$UMDIFF_ROOT/_deps${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
