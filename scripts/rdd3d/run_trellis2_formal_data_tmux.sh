#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SPLIT="${RDD3D_SPLIT:-train}"
RESOLUTION="${RDD3D_OVOXEL_RESOLUTION:-1024}"
SESSION="rdd3d_data_${SPLIT}_${RESOLUTION}"
LOG="${ROOT}/logs/${SESSION}.log"

mkdir -p "${ROOT}/logs"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION}" >&2
    exit 2
fi

TMUX_ENV=()
for NAME in RDD3D_SPLIT RDD3D_SOURCE RDD3D_SUBSET_COUNT RDD3D_SUBSET_SEED \
    RDD3D_MIN_AESTHETIC_SCORE RDD3D_OVOXEL_RESOLUTION RDD3D_DATA_ROOT \
    RDD3D_REPORT_ROOT RDD3D_EXCLUDE_METADATA RDD3D_DOWNLOAD_WORLD_SIZE \
    RDD3D_DOWNLOAD_RANK RDD3D_RENDER_COND RDD3D_NUM_VIEWS RDD3D_MAX_WORKERS \
    RDD3D_GPU RDD3D_RENDER_GPUS; do
    if [[ -n "${!NAME:-}" ]]; then
        TMUX_ENV+=(-e "${NAME}=${!NAME}")
    fi
done

tmux new-session -d -s "${SESSION}" "${TMUX_ENV[@]}" \
    "bash -lc 'cd \"${ROOT}\" && bash scripts/rdd3d/prepare_trellis2_formal_split.sh 2>&1 | tee \"${LOG}\"'"

echo "Started ${SESSION}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Log:    tail -f ${LOG}"
