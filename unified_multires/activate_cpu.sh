# Source this file from Bash. It changes no shared virtual environment.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/activate.sh"
export CUDA_VISIBLE_DEVICES=""
export UMDIFF_CPU_ONLY=1
