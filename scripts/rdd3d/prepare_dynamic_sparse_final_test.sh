#!/usr/bin/env bash
set -euo pipefail

# Freeze a test set that is disjoint from pilot, formal train, and the 32-asset
# development held-out split.  Building links/conditions is allowed before the
# final model is chosen, but learned metrics on this split must be run only once.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${RDD3D_DYNAMIC_PYTHON:-${ROOT}/.venv_rdd3d/bin/python}"
DATA_ROOT="${RDD3D_FORMAL_DATA_ROOT:-${ROOT}/datasets/rdd3d_formal/ObjaverseXL_github_train_256}"
SOURCE_LATENTS="${RDD3D_FORMAL_LATENTS:-${DATA_ROOT}/shape_latents/shape_enc_next_dc_f16c32_fp16_512}"
RENDERS="${RDD3D_FORMAL_RENDERS:-${DATA_ROOT}/renders_cond}"
DEV_SPLIT="${RDD3D_FORMAL_SPLIT_ROOT:-${DATA_ROOT}/rdd_dynamic_split_96_32_seed98317}"
PILOT_LATENTS="${RDD3D_PILOT_LATENTS:-${ROOT}/datasets/rdd3d_pilot/ObjaverseXL_github/shape_latents/shape_enc_next_dc_f16c32_fp16_512}"
TEST_COUNT="${RDD3D_FINAL_TEST_COUNT:-64}"
TEST_SEED="${RDD3D_FINAL_TEST_SEED:-20260812}"
FINAL_ROOT="${RDD3D_FINAL_TEST_ROOT:-${DATA_ROOT}/rdd_dynamic_final_test_${TEST_COUNT}_seed${TEST_SEED}}"
FINAL_LATENTS="${FINAL_ROOT}/heldout_latents"
CONDITIONS="${FINAL_ROOT}/conditions_dinov2_vits14"

for REQUIRED in "${PYTHON}" "${SOURCE_LATENTS}" "${RENDERS}" \
    "${DEV_SPLIT}/train_latents" "${DEV_SPLIT}/heldout_latents"; do
    if [[ ! -e "${REQUIRED}" ]]; then
        echo "Missing final-test input: ${REQUIRED}" >&2
        exit 2
    fi
done
cd "${ROOT}"
export PYTHONPATH="${ROOT}/RDD_DiT_3D:${PYTHONPATH:-}"

SPLIT_ARGS=(
    --latent-dir "${SOURCE_LATENTS}" --render-dir "${RENDERS}"
    --output-root "${FINAL_ROOT}" --train-count 0
    --heldout-count "${TEST_COUNT}" --seed "${TEST_SEED}"
    --exclude-latent-dir "${DEV_SPLIT}/train_latents"
    --exclude-latent-dir "${DEV_SPLIT}/heldout_latents"
)
if [[ -d "${PILOT_LATENTS}" ]]; then
    SPLIT_ARGS+=(--exclude-latent-dir "${PILOT_LATENTS}")
fi
"${PYTHON}" scripts/rdd3d/build_dynamic_sparse_formal_split.py "${SPLIT_ARGS[@]}"

CUDA_VISIBLE_DEVICES="${RDD3D_FINAL_TEST_GPU:-0}" "${PYTHON}" \
    RDD_DiT_3D/precompute_dynamic_sparse_conditions.py \
    --render-dir "${RENDERS}" --latent-dir "${FINAL_LATENTS}" \
    --output "${CONDITIONS}" --model dinov2_vits14

"${PYTHON}" RDD_DiT_3D/analyze_dynamic_sparse_rdd.py \
    --latent-dir "${FINAL_LATENTS}" --limit "${TEST_COUNT}" \
    --fine-resolution 32 --resolutions 8 16 32 --steps 8 12 20 \
    --output "${FINAL_ROOT}/oracle_sparsity.json"

echo "RDD3D_FINAL_TEST_FROZEN"
echo "Manifest:   ${FINAL_ROOT}/manifest.json"
echo "Latents:    ${FINAL_LATENTS}"
echo "Conditions: ${CONDITIONS}"
echo "Do not run learned evaluation until the final method/config is frozen."
