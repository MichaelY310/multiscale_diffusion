#!/usr/bin/env bash
set -euo pipefail

# Paired TRELLIS decode for the fixed stage-interface strict-F2 operating point.
# This intentionally matches the formal gate: 4 held-out assets per fixed seed,
# 512 decode resolution, 50K surface samples.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${RDD3D_DYNAMIC_PYTHON:-${ROOT}/.venv_rdd3d/bin/python}"
DATA_ROOT="${RDD3D_FORMAL_DATA_ROOT:-${ROOT}/datasets/rdd3d_formal/ObjaverseXL_github_train_256}"
SPLIT_ROOT="${RDD3D_FORMAL_SPLIT_ROOT:-${DATA_ROOT}/rdd_dynamic_split_96_32_seed98317}"
HELDOUT_LATENTS="${SPLIT_ROOT}/heldout_latents"
RUN_NAME="${RDD3D_FORMAL_RUN_NAME:-dynamic_sparse_formal_96train_32heldout_w384d12}"
OUTPUT_ROOT="${ROOT}/results_rdd3d/${RUN_NAME}"
EVAL_ROOT="${ROOT}/eval_results/rdd3d_dynamic_sparse/${RUN_NAME}"
TAG="${RDD3D_INTERFACE_MESH_TAG:-interface19coarse_refine34k_closed_loop_f2_cal_step0029000}"
LIMIT="${RDD3D_INTERFACE_MESH_LIMIT:-4}"

for REQUIRED in "${PYTHON}" "${HELDOUT_LATENTS}"; do
    if [[ ! -e "${REQUIRED}" ]]; then
        echo "Missing mesh-evaluation input: ${REQUIRED}" >&2
        exit 2
    fi
done
mkdir -p "${EVAL_ROOT}" "${ROOT}/logs"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/RDD_DiT_3D:${ROOT}/third_party/TRELLIS.2:${PYTHONPATH:-}"

pids=()
for SEED_GPU in "314159:0" "271828:1"; do
    SEED="${SEED_GPU%%:*}"
    GPU="${SEED_GPU##*:}"
    SAMPLES="${OUTPUT_ROOT}/samples_${TAG}_seed${SEED}"
    OUTPUT="${EVAL_ROOT}/${TAG}_seed${SEED}_mesh.json"
    if [[ ! -s "${SAMPLES}/manifest.json" ]]; then
        echo "Missing sample manifest: ${SAMPLES}/manifest.json" >&2
        exit 2
    fi
    if [[ ! -s "${OUTPUT}" ]]; then
        SPARSE_CONV_BACKEND=flex_gemm CUDA_VISIBLE_DEVICES="${GPU}" \
            "${PYTHON}" RDD_DiT_3D/evaluate_dynamic_sparse_decode.py \
            --prediction-dir "${SAMPLES}" --ground-truth-dir "${HELDOUT_LATENTS}" \
            --output "${OUTPUT}" --limit "${LIMIT}" --resolution 512 \
            --sample-points 50000 \
            >"${ROOT}/logs/${RUN_NAME}_${TAG}_mesh_seed${SEED}.log" 2>&1 &
        pids+=("$!")
    fi
done
for pid in "${pids[@]}"; do
    wait "${pid}"
done

"${PYTHON}" RDD_DiT_3D/summarize_dynamic_sparse_formal.py \
    --output-root "${OUTPUT_ROOT}" --eval-root "${EVAL_ROOT}" \
    --output-json "${EVAL_ROOT}/formal_summary.json" \
    --output-markdown "${EVAL_ROOT}/formal_summary.md"
echo "RDD3D_STAGE_INTERFACE_MESH_COMPLETE"
