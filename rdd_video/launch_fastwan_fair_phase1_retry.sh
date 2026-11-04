#!/usr/bin/env bash
set -euo pipefail
ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
SESSION="rdd_fair_reuse_projected"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "session already exists: ${SESSION}" >&2
  exit 1
fi
tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && env GPU='1' NAME='temporal_vsa20_reuse2_projected' FACTORS='1,2;1,2;1,1' ATTENTION_MODE='vsa' VSA_SPARSITY='0.8' REUSE_EVERY='2' REUSE_SCOPE='projected' WARMUP_RUNS='1' TIMED_RUNS='5' bash '${ROOT}/rdd_video/run_fastwan_fair_worker.sh'"
echo "restarted ${SESSION} on GPU 1"
