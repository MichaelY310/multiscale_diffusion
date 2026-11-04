#!/usr/bin/env bash
set -euo pipefail
unified_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$unified_root/activate_cpu.sh"
exec "$UMDIFF_PYTHON" "$unified_root/tools/run_supported_reproduction_checks.py" "$@"
