#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 RDD3D_ONPOLICY_THRESHOLD_SHARD=0 \
    bash scripts/rdd3d/run_dynamic_sparse_onpolicy_threshold_shard.sh \
    >logs/rdd3d_onpolicy_threshold_19k.log 2>&1 &
PID0=$!
CUDA_VISIBLE_DEVICES=1 RDD3D_ONPOLICY_THRESHOLD_SHARD=1 \
    bash scripts/rdd3d/run_dynamic_sparse_onpolicy_threshold_shard.sh \
    >logs/rdd3d_onpolicy_threshold_20k.log 2>&1 &
PID1=$!
wait "${PID0}"
wait "${PID1}"
.venv_rdd3d/bin/python RDD_DiT_3D/summarize_dynamic_sparse_ablations.py \
    --sample-root results_rdd3d/dynamic_sparse_onpolicy_threshold_seed314159 \
    --eval-root eval_results/rdd3d_dynamic_sparse/onpolicy_threshold_seed314159 \
    --output eval_results/rdd3d_dynamic_sparse/onpolicy_threshold_seed314159/summary.json
echo "RDD3D_ONPOLICY_THRESHOLD_COMPLETE"
