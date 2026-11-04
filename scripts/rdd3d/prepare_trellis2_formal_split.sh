#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${ROOT}/.venv_rdd3d/bin/python"
SPLIT="${RDD3D_SPLIT:-train}"
SOURCE="${RDD3D_SOURCE:-sketchfab}"
COUNT="${RDD3D_SUBSET_COUNT:-10000}"
SEED="${RDD3D_SUBSET_SEED:-3407}"
MIN_SCORE="${RDD3D_MIN_AESTHETIC_SCORE:-5.5}"
RESOLUTION="${RDD3D_OVOXEL_RESOLUTION:-1024}"
DATA_ROOT="${RDD3D_DATA_ROOT:-${ROOT}/datasets/rdd3d_formal/ObjaverseXL_${SOURCE}_${SPLIT}_${COUNT}}"
REPORT_ROOT="${RDD3D_REPORT_ROOT:-${ROOT}/eval_results/rdd3d_formal/${SPLIT}_${RESOLUTION}}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing ${PYTHON}; run scripts/rdd3d/setup_trellis2_pilot.sh first." >&2
    exit 2
fi

SUBSET_ARGS=(
    --output-root "${DATA_ROOT}"
    --source "${SOURCE}"
    --count "${COUNT}"
    --seed "${SEED}"
    --split "${SPLIT}"
    --min-aesthetic-score "${MIN_SCORE}"
)
if [[ -n "${RDD3D_EXCLUDE_METADATA:-}" ]]; then
    SUBSET_ARGS+=(--exclude "${RDD3D_EXCLUDE_METADATA}")
fi
if [[ ! -f "${DATA_ROOT}/metadata.csv" ]]; then
    "${PYTHON}" "${ROOT}/scripts/rdd3d/create_objaversexl_subset.py" "${SUBSET_ARGS[@]}"
fi

export RDD3D_DATA_ROOT="${DATA_ROOT}"
export RDD3D_REPORT_ROOT="${REPORT_ROOT}"
export RDD3D_SOURCE="${SOURCE}"
export RDD3D_OVOXEL_RESOLUTION="${RESOLUTION}"
export RDD3D_DOWNLOAD_WORLD_SIZE="${RDD3D_DOWNLOAD_WORLD_SIZE:-1}"
export RDD3D_DOWNLOAD_RANK="${RDD3D_DOWNLOAD_RANK:-0}"
export RDD3D_RENDER_COND="${RDD3D_RENDER_COND:-1}"
export RDD3D_NUM_VIEWS="${RDD3D_NUM_VIEWS:-16}"
export RDD3D_MAX_WORKERS="${RDD3D_MAX_WORKERS:-8}"

bash "${ROOT}/scripts/rdd3d/prepare_trellis2_pilot_data.sh"
