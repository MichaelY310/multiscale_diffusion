#!/usr/bin/env bash
set -Eeuo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
ROOT=/home/yuheng/rdd_experiments/dmd_cascade_reuse_20260915
FAMILY=${1:?high_heavy or balanced_plus}
case "$FAMILY" in
  high_heavy) export SPEED_GATE="$ROOT/video_spatial_speed/speed_gate.json" ;;
  balanced_plus) export SPEED_GATE="$ROOT/video_spatial_plus_speed/speed_gate.json" ;;
  *) exit 2 ;;
esac
mkdir -p "$ROOT/$FAMILY"
exec > >(tee -a "$ROOT/$FAMILY/queue.log") 2>&1
while [[ ! -s "$SPEED_GATE" || ! -s "$ROOT/smoke4_${FAMILY}/smoke_result.json" ]]; do
  echo "Waiting for exact spatial >=20% speed gate and four-GPU smoke: $FAMILY"
  sleep 20
done
.venv-rdd-eval/bin/python -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["complete"] and d["state"]=="smoke_passed"' "$ROOT/smoke4_${FAMILY}/smoke_result.json"
exec bash 2d/exp/run_dmd_cascade_reuse_20260915.sh "$FAMILY" train
