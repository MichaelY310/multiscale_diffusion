#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
mkdir -p "${ROOT}/logs"
cd "${ROOT}"

CUDA_VISIBLE_DEVICES=0 RDD3D_THRESHOLD_SHARD=0 \
    bash scripts/rdd3d/run_dynamic_sparse_threshold_ablation_shard.sh \
    >logs/rdd3d_threshold_ablation_shard0.log 2>&1 &
PID0=$!
CUDA_VISIBLE_DEVICES=1 RDD3D_THRESHOLD_SHARD=1 \
    bash scripts/rdd3d/run_dynamic_sparse_threshold_ablation_shard.sh \
    >logs/rdd3d_threshold_ablation_shard1.log 2>&1 &
PID1=$!

STATUS=0
wait "${PID0}" || STATUS=$?
wait "${PID1}" || STATUS=$?
if (( STATUS != 0 )); then
    echo "threshold ablation shard failed: status=${STATUS}" >&2
    tail -n 40 logs/rdd3d_threshold_ablation_shard0.log || true
    tail -n 40 logs/rdd3d_threshold_ablation_shard1.log || true
    exit "${STATUS}"
fi

.venv_rdd3d/bin/python RDD_DiT_3D/summarize_dynamic_sparse_ablations.py \
    --sample-root results_rdd3d/dynamic_sparse_15k_threshold_ablations_seed20260811 \
    --eval-root eval_results/rdd3d_dynamic_sparse/15k_threshold_ablations_seed20260811 \
    --output eval_results/rdd3d_dynamic_sparse/15k_threshold_ablations_seed20260811/summary.json

echo "RDD3D_THRESHOLD_ABLATION_COMPLETE"
