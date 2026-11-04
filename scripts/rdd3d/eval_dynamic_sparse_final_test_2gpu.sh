#!/usr/bin/env bash
set -euo pipefail

# One-shot evaluation on the frozen 64-asset final test.  Method identities,
# checkpoints, train-only thresholds, seeds, sampler NFE, and pruning rules are
# fixed below.  Do not add methods or tune thresholds after inspecting output.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${RDD3D_DYNAMIC_PYTHON:-${ROOT}/.venv_rdd3d/bin/python}"
DATA_ROOT="${RDD3D_FORMAL_DATA_ROOT:-${ROOT}/datasets/rdd3d_formal/ObjaverseXL_github_train_256}"
TEST_COUNT="${RDD3D_FINAL_TEST_COUNT:-64}"
TEST_SEED="${RDD3D_FINAL_TEST_SEED:-20260812}"
FINAL_ROOT="${RDD3D_FINAL_TEST_ROOT:-${DATA_ROOT}/rdd_dynamic_final_test_${TEST_COUNT}_seed${TEST_SEED}}"
FINAL_LATENTS="${FINAL_ROOT}/heldout_latents"
CONDITIONS="${FINAL_ROOT}/conditions_dinov2_vits14"

BASE_RUN="${RDD3D_FORMAL_BASE_RUN_NAME:-dynamic_sparse_formal_96train_32heldout_w384d12}"
DEV_OUTPUT="${ROOT}/results_rdd3d/${BASE_RUN}"
DEV_EVAL="${ROOT}/eval_results/rdd3d_dynamic_sparse/${BASE_RUN}"
HEAD_RUN="${BASE_RUN}_endpoint_head"
HEAD_OUTPUT="${ROOT}/results_rdd3d/${HEAD_RUN}"
HEAD_EVAL="${ROOT}/eval_results/rdd3d_dynamic_sparse/${HEAD_RUN}"

ZERO_CHECKPOINT="${ROOT}/results_rdd3d/dynamic_sparse_32asset_onpolicy_ft_w384d12_2gpu/checkpoints/step0019000.pt"
ZERO_THRESHOLDS="${DEV_EVAL}/zeroshot_step0019000_thresholds.json"
ONPOLICY_CHECKPOINT="${DEV_OUTPUT}/onpolicy_replay/checkpoints/step0029000.pt"
ONPOLICY_THRESHOLDS="${DEV_EVAL}/onpolicy_step0029000_clean_thresholds.json"
SELECTED_CHECKPOINT="${HEAD_OUTPUT}/endpoint_head/checkpoint.pt"
SELECTED_THRESHOLDS="${HEAD_EVAL}/endpoint_head_closed_loop_f2_thresholds.json"

RUN_NAME="dynamic_sparse_final_test_${TEST_COUNT}_seed${TEST_SEED}"
OUTPUT_ROOT="${ROOT}/results_rdd3d/${RUN_NAME}"
EVAL_ROOT="${ROOT}/eval_results/rdd3d_dynamic_sparse/${RUN_NAME}"
MESH_LIMIT="${RDD3D_FINAL_MESH_LIMIT:-16}"
BENCHMARK_LIMIT="${RDD3D_FINAL_BENCHMARK_LIMIT:-16}"
SEEDS=(314159 271828)

for REQUIRED in "${PYTHON}" "${FINAL_ROOT}/manifest.json" "${FINAL_ROOT}/oracle_sparsity.json" \
    "${FINAL_LATENTS}" "${CONDITIONS}" "${ZERO_CHECKPOINT}" "${ZERO_THRESHOLDS}" \
    "${ONPOLICY_CHECKPOINT}" "${ONPOLICY_THRESHOLDS}" \
    "${SELECTED_CHECKPOINT}" "${SELECTED_THRESHOLDS}"; do
    if [[ ! -e "${REQUIRED}" ]]; then
        echo "Missing frozen final-test input: ${REQUIRED}" >&2
        exit 2
    fi
done
mkdir -p "${OUTPUT_ROOT}" "${EVAL_ROOT}" "${ROOT}/logs"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/RDD_DiT_3D:${ROOT}/third_party/TRELLIS.2:${PYTHONPATH:-}"

sample_one() {
    local tag="$1"
    local checkpoint="$2"
    local thresholds="$3"
    local mode="$4"
    local seed="$5"
    local gpu="$6"
    local samples="${OUTPUT_ROOT}/samples_${tag}_seed${seed}"
    local report="${EVAL_ROOT}/${tag}_seed${seed}_heldout_support.json"
    if [[ ! -s "${samples}/manifest.json" ]]; then
        CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
            --checkpoint "${checkpoint}" --latent-dir "${FINAL_LATENTS}" \
            --condition-dir "${CONDITIONS}" --thresholds "${thresholds}" \
            --output-dir "${samples}" --steps 8 12 20 \
            --mass-thresholds 0.05 0.10 0.25 --pruning-mode "${mode}" \
            --limit "${TEST_COUNT}" --seed "${seed}" --weights model \
            >"${ROOT}/logs/${RUN_NAME}_${tag}_seed${seed}.log" 2>&1
    fi
    if [[ ! -s "${report}" ]]; then
        "${PYTHON}" RDD_DiT_3D/evaluate_dynamic_sparse_predictions.py \
            --prediction-dir "${samples}" --ground-truth-dir "${FINAL_LATENTS}" \
            --output "${report}" --resolution 32 --stage-resolutions 8 16 32
    fi
}

evaluate_method() {
    local tag="$1"
    local checkpoint="$2"
    local thresholds="$3"
    local mode="$4"
    local pids=()
    sample_one "${tag}" "${checkpoint}" "${thresholds}" "${mode}" "${SEEDS[0]}" 0 &
    pids+=("$!")
    sample_one "${tag}" "${checkpoint}" "${thresholds}" "${mode}" "${SEEDS[1]}" 1 &
    pids+=("$!")
    for pid in "${pids[@]}"; do
        wait "${pid}"
    done
}

echo "===== Final test 1/5: frozen zero-shot 19K clean/OR ====="
evaluate_method zeroshot19_clean_or "${ZERO_CHECKPOINT}" "${ZERO_THRESHOLDS}" or

echo "===== Final test 2/5: frozen on-policy 29K clean/OR ====="
evaluate_method onpolicy29_clean_or "${ONPOLICY_CHECKPOINT}" "${ONPOLICY_THRESHOLDS}" or

echo "===== Final test 3/5: selected 29K endpoint-head strict-F2 ====="
SELECTED_TAG="onpolicy29_endpoint_head_strict_f2"
evaluate_method "${SELECTED_TAG}" "${SELECTED_CHECKPOINT}" "${SELECTED_THRESHOLDS}" probability

echo "===== Final test 4/5: selected model force-dense runtime control ====="
DENSE_TAG="onpolicy29_endpoint_head_force_dense"
pids=()
for SEED_GPU in "${SEEDS[0]}:0" "${SEEDS[1]}:1"; do
    SEED="${SEED_GPU%%:*}"
    GPU="${SEED_GPU##*:}"
    SAMPLES="${OUTPUT_ROOT}/samples_${DENSE_TAG}_seed${SEED}"
    if [[ ! -s "${SAMPLES}/manifest.json" ]]; then
        CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
            --checkpoint "${SELECTED_CHECKPOINT}" --latent-dir "${FINAL_LATENTS}" \
            --condition-dir "${CONDITIONS}" --thresholds "${SELECTED_THRESHOLDS}" \
            --output-dir "${SAMPLES}" --steps 8 12 20 \
            --mass-thresholds 0.05 0.10 0.25 --pruning-mode probability \
            --force-keep-resolutions 8 16 32 --limit "${BENCHMARK_LIMIT}" \
            --seed "${SEED}" --weights model \
            >"${ROOT}/logs/${RUN_NAME}_${DENSE_TAG}_seed${SEED}.log" 2>&1 &
        pids+=("$!")
    fi
done
for pid in "${pids[@]}"; do
    wait "${pid}"
done

"${PYTHON}" RDD_DiT_3D/summarize_dynamic_sparse_runtime.py \
    --sparse-manifests \
        "${OUTPUT_ROOT}/samples_${SELECTED_TAG}_seed${SEEDS[0]}/manifest.json" \
        "${OUTPUT_ROOT}/samples_${SELECTED_TAG}_seed${SEEDS[1]}/manifest.json" \
    --dense-manifests \
        "${OUTPUT_ROOT}/samples_${DENSE_TAG}_seed${SEEDS[0]}/manifest.json" \
        "${OUTPUT_ROOT}/samples_${DENSE_TAG}_seed${SEEDS[1]}/manifest.json" \
    --total-limit "${BENCHMARK_LIMIT}" --warmup-per-manifest 1 \
    --output-json "${EVAL_ROOT}/runtime_benchmark.json" \
    --output-markdown "${EVAL_ROOT}/runtime_benchmark.md"

echo "===== Final test 5/5: paired selected-method TRELLIS mesh ====="
pids=()
for SEED_GPU in "${SEEDS[0]}:0" "${SEEDS[1]}:1"; do
    SEED="${SEED_GPU%%:*}"
    GPU="${SEED_GPU##*:}"
    SAMPLES="${OUTPUT_ROOT}/samples_${SELECTED_TAG}_seed${SEED}"
    REPORT="${EVAL_ROOT}/${SELECTED_TAG}_seed${SEED}_mesh.json"
    if [[ ! -s "${REPORT}" ]]; then
        SPARSE_CONV_BACKEND=flex_gemm CUDA_VISIBLE_DEVICES="${GPU}" \
            "${PYTHON}" RDD_DiT_3D/evaluate_dynamic_sparse_decode.py \
            --prediction-dir "${SAMPLES}" --ground-truth-dir "${FINAL_LATENTS}" \
            --output "${REPORT}" --limit "${MESH_LIMIT}" --resolution 512 \
            --sample-points 50000 \
            >"${ROOT}/logs/${RUN_NAME}_${SELECTED_TAG}_mesh_seed${SEED}.log" 2>&1 &
        pids+=("$!")
    fi
done
for pid in "${pids[@]}"; do
    wait "${pid}"
done

"${PYTHON}" RDD_DiT_3D/summarize_dynamic_sparse_formal.py \
    --output-root "${OUTPUT_ROOT}" --eval-root "${EVAL_ROOT}" \
    --output-json "${EVAL_ROOT}/final_summary.json" \
    --output-markdown "${EVAL_ROOT}/final_summary.md"
echo "RDD3D_FINAL_TEST_EVALUATION_COMPLETE"
echo "Summary: ${EVAL_ROOT}/final_summary.md"
