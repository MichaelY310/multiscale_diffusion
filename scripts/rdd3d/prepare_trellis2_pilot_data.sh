#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRELLIS="${ROOT}/third_party/TRELLIS.2"
PYTHON="${ROOT}/.venv_rdd3d/bin/python"
DATA_ROOT="${RDD3D_DATA_ROOT:-${ROOT}/datasets/rdd3d_pilot/ObjaverseXL_github}"
REPORT_ROOT="${RDD3D_REPORT_ROOT:-${ROOT}/eval_results/rdd3d_pilot}"
SOURCE="${RDD3D_SOURCE:-github}"
RESOLUTION="${RDD3D_OVOXEL_RESOLUTION:-512}"
LATENT_RESOLUTION=$((RESOLUTION / 16))
MIN_LATENT_RESOLUTION="${RDD3D_MIN_LATENT_RESOLUTION:-$((LATENT_RESOLUTION / 4))}"
WORLD_SIZE="${RDD3D_DOWNLOAD_WORLD_SIZE:-8000}"
RANK="${RDD3D_DOWNLOAD_RANK:-0}"
GPU="${RDD3D_GPU:-4}"
RENDER_GPUS="${RDD3D_RENDER_GPUS:-${GPU}}"
MAX_WORKERS="${RDD3D_MAX_WORKERS:-8}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing ${PYTHON}; run scripts/rdd3d/setup_trellis2_pilot.sh first." >&2
    exit 2
fi

mkdir -p "${DATA_ROOT}" "${REPORT_ROOT}" "${ROOT}/logs"
cd "${TRELLIS}"
export PYTHONPATH="${ROOT}/RDD_DiT_3D/trellis2_data_adapters:${PYTHONPATH:-}"

# The official dumper expects this exact /tmp location.  Downloading Blender is
# user-space and does not require the script's attempted sudo apt invocation.
BLENDER_DIR=/tmp/blender-4.5.1-linux-x64
if [[ ! -x "${BLENDER_DIR}/blender" ]]; then
    BLENDER_TAR=/tmp/blender-4.5.1-linux-x64.tar.xz
    wget -c https://ftp.halifax.rwth-aachen.de/blender/release/Blender4.5/blender-4.5.1-linux-x64.tar.xz \
        -O "${BLENDER_TAR}"
    tar -xf "${BLENDER_TAR}" -C /tmp
fi

# render_cond.py is pinned to Blender 3.0.1, independently of the 4.5 mesh
# dumper above.  Install it in user-writable /tmp so the upstream helper never
# falls through to its sudo apt-get branch on a compute server.
COND_BLENDER_DIR=/tmp/blender-3.0.1-linux-x64
if [[ "${RDD3D_RENDER_COND:-0}" == 1 && ! -x "${COND_BLENDER_DIR}/blender" ]]; then
    COND_BLENDER_TAR=/tmp/blender-3.0.1-linux-x64.tar.xz
    wget -c https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz \
        -O "${COND_BLENDER_TAR}"
    tar -xf "${COND_BLENDER_TAR}" -C /tmp
fi

"${PYTHON}" data_toolkit/build_metadata.py ObjaverseXL --source "${SOURCE}" --root "${DATA_ROOT}"
DOWNLOAD_ARGS=(
    ObjaverseXL --source "${SOURCE}" --root "${DATA_ROOT}"
    --rank "${RANK}" --world_size "${WORLD_SIZE}"
)
if [[ "${RDD3D_EXISTING_ONLY:-0}" == 1 ]]; then
    if [[ "${SOURCE}" != github ]]; then
        echo "RDD3D_EXISTING_ONLY is only supported for the GitHub source." >&2
        exit 2
    fi
    DOWNLOAD_ARGS+=(--existing_only)
fi
"${PYTHON}" data_toolkit/download.py "${DOWNLOAD_ARGS[@]}"
"${PYTHON}" data_toolkit/build_metadata.py ObjaverseXL --source "${SOURCE}" --root "${DATA_ROOT}"
"${PYTHON}" data_toolkit/dump_mesh.py ObjaverseXL --root "${DATA_ROOT}" --max_workers "${MAX_WORKERS}"
"${PYTHON}" data_toolkit/build_metadata.py ObjaverseXL --source "${SOURCE}" --root "${DATA_ROOT}"
"${PYTHON}" data_toolkit/dual_grid.py ObjaverseXL --root "${DATA_ROOT}" \
    --resolution "${RESOLUTION}" --max_workers "${MAX_WORKERS}"
"${PYTHON}" data_toolkit/build_metadata.py ObjaverseXL --source "${SOURCE}" --root "${DATA_ROOT}"

SPARSE_CONV_BACKEND=flex_gemm CUDA_VISIBLE_DEVICES="${GPU}" \
    "${PYTHON}" data_toolkit/encode_shape_latent.py --root "${DATA_ROOT}" \
    --resolution "${RESOLUTION}"
"${PYTHON}" data_toolkit/build_metadata.py ObjaverseXL --source "${SOURCE}" --root "${DATA_ROOT}"

if [[ "${RDD3D_RENDER_COND:-0}" == 1 ]]; then
    IFS=',' read -r -a RENDER_GPU_ARRAY <<<"${RENDER_GPUS}"
    RENDER_WORLD_SIZE="${#RENDER_GPU_ARRAY[@]}"
    if (( RENDER_WORLD_SIZE > 1 )); then
        # Blender enables every CUDA device it can see.  Exposing a comma-
        # separated GPU list to every worker therefore replicates each worker
        # on every GPU and can exhaust VRAM.  Shard the deterministic metadata
        # list and expose exactly one physical GPU to each rank instead.
        RENDER_WORKERS_PER_RANK=$(( (MAX_WORKERS + RENDER_WORLD_SIZE - 1) / RENDER_WORLD_SIZE ))
        RENDER_PIDS=()
        for RENDER_RANK in "${!RENDER_GPU_ARRAY[@]}"; do
            CUDA_VISIBLE_DEVICES="${RENDER_GPU_ARRAY[${RENDER_RANK}]}" \
                "${PYTHON}" data_toolkit/render_cond.py ObjaverseXL \
                --root "${DATA_ROOT}" \
                --num_cond_views "${RDD3D_NUM_VIEWS:-16}" \
                --rank "${RENDER_RANK}" --world_size "${RENDER_WORLD_SIZE}" \
                --max_workers "${RENDER_WORKERS_PER_RANK}" &
            RENDER_PIDS+=("$!")
        done
        RENDER_FAILED=0
        for RENDER_PID in "${RENDER_PIDS[@]}"; do
            wait "${RENDER_PID}" || RENDER_FAILED=1
        done
        if (( RENDER_FAILED )); then
            echo "At least one conditional-render rank failed." >&2
            exit 1
        fi
    else
        CUDA_VISIBLE_DEVICES="${RENDER_GPU_ARRAY[0]}" \
            "${PYTHON}" data_toolkit/render_cond.py ObjaverseXL --root "${DATA_ROOT}" \
            --num_cond_views "${RDD3D_NUM_VIEWS:-16}" --max_workers "${MAX_WORKERS}"
    fi
    "${PYTHON}" data_toolkit/build_metadata.py ObjaverseXL --source "${SOURCE}" --root "${DATA_ROOT}"
fi

cd "${ROOT}"
"${PYTHON}" RDD_DiT_3D/analyze_ovoxel_sparsity.py \
    "${DATA_ROOT}/dual_grid_${RESOLUTION}" \
    --resolution "${RESOLUTION}" --min-resolution 8 \
    --output "${REPORT_ROOT}/ovoxel_${RESOLUTION}_sparsity.json"
"${PYTHON}" RDD_DiT_3D/analyze_slat_pyramid.py \
    "${DATA_ROOT}/shape_latents/shape_enc_next_dc_f16c32_fp16_${RESOLUTION}" \
    --resolution "${LATENT_RESOLUTION}" --min-resolution "${MIN_LATENT_RESOLUTION}" \
    --step-allocation ${RDD3D_STEP_ALLOCATION:-8 12 20} \
    --output "${REPORT_ROOT}/slat_${LATENT_RESOLUTION}_pyramid.json"

echo "Pilot data: ${DATA_ROOT}"
echo "Reports:    ${REPORT_ROOT}"
