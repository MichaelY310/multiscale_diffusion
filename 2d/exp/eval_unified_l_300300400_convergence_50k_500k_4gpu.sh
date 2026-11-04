#!/usr/bin/env bash
# Public entry point for the complete Unified-L 50K--500K convergence curve.

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/eval_unified_l_300300400_convergence_50k_250k_4gpu.sh" "$@"
