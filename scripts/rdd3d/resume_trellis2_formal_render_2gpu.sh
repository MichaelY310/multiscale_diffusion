#!/usr/bin/env bash
set -euo pipefail

# Resume only the interrupted conditioning-render tail of formal preprocessing.
# Existing transforms.json files are detected and skipped by TRELLIS.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRELLIS="${ROOT}/third_party/TRELLIS.2"
PYTHON="${ROOT}/.venv_rdd3d/bin/python"
DATA_ROOT="${RDD3D_DATA_ROOT:-${ROOT}/datasets/rdd3d_formal/ObjaverseXL_github_train_256}"
REPORT_ROOT="${RDD3D_REPORT_ROOT:-${ROOT}/eval_results/rdd3d_formal/train_512}"
RENDER_GPUS="${RDD3D_RENDER_GPUS:-0,1}"
MAX_WORKERS="${RDD3D_RENDER_MAX_WORKERS:-2}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing ${PYTHON}" >&2
    exit 2
fi
for REQUIRED in \
    "${DATA_ROOT}/metadata.csv" \
    "${DATA_ROOT}/dual_grid_512" \
    "${DATA_ROOT}/shape_latents/shape_enc_next_dc_f16c32_fp16_512"; do
    if [[ ! -e "${REQUIRED}" ]]; then
        echo "Missing prerequisite: ${REQUIRED}" >&2
        exit 2
    fi
done

mkdir -p "${REPORT_ROOT}" "${ROOT}/logs"
cd "${TRELLIS}"
export PYTHONPATH="${ROOT}/RDD_DiT_3D/trellis2_data_adapters:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES="${RENDER_GPUS}" \
    "${PYTHON}" data_toolkit/render_cond.py ObjaverseXL --root "${DATA_ROOT}" \
    --num_cond_views 1 --max_workers "${MAX_WORKERS}"
"${PYTHON}" data_toolkit/build_metadata.py ObjaverseXL \
    --source github --root "${DATA_ROOT}"

cd "${ROOT}"
"${PYTHON}" RDD_DiT_3D/analyze_ovoxel_sparsity.py \
    "${DATA_ROOT}/dual_grid_512" --resolution 512 --min-resolution 8 \
    --output "${REPORT_ROOT}/ovoxel_512_sparsity.json"
"${PYTHON}" RDD_DiT_3D/analyze_slat_pyramid.py \
    "${DATA_ROOT}/shape_latents/shape_enc_next_dc_f16c32_fp16_512" \
    --resolution 32 --min-resolution 8 --step-allocation 8 12 20 \
    --output "${REPORT_ROOT}/slat_32_pyramid.json"

echo "RDD3D_FORMAL_RENDER_RESUME_COMPLETE"
echo "Data:    ${DATA_ROOT}"
echo "Reports: ${REPORT_ROOT}"
