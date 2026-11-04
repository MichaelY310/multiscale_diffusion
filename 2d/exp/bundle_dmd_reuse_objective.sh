#!/usr/bin/env bash
# Evidence only: never copies model/optimizer checkpoints or changes source data.
set -Eeuo pipefail
cd /home/yuheng/rdd_experiments/dmd_reuse_objective_20260915
find control dmd_lpips smoke_control smoke_dmd_lpips -type f \
  \( -name '*.json' -o -name '*.jsonl' -o -name 'comparison_first16.jpg' -o -name 'launcher.log' \) \
  -print0 | tar --null -T - -czf monitor_snapshot.tar.gz
