#!/usr/bin/env bash
# Predeclared control and fresh256-pair validation of split3+3, not more epochs.
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
ROOT=/home/yuheng/rdd_experiments/csra_skip_layout_20260915
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
exec 9>"$ROOT/split_followup.lock"
flock -n 9 || { echo 'This exact split follow-up already runs'; exit 1; }
for name in front0_6 front3_6 front6_6 split3_3; do
 test -f "$ROOT/training/${name}_aligned/complete.json"
done
for gpu in 0 1 2 3; do test -f "$ROOT/layout_quality/shard_$gpu/metrics.json"; done
check_memory() {
 local m
 mapfile -t mem < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
 [[ ${#mem[@]} == 4 ]] || return 1
 for m in "${mem[@]}"; do ((m>=28672)) || return 1; done
}
until check_memory; do echo 'Waiting28GiB free on0-3; GPUs4-7 remain reserved for others'; sleep 30; done
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
 2d/attention_analysis/train_csra_layout_pilot.py --root "$ROOT" --layout split3_3 --source zero --steps 2 --smoke \
 2>&1 | tee -a "$ROOT/train_smoke_split3_3_zero.log"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
 2d/attention_analysis/train_csra_layout_pilot.py --root "$ROOT" --layout split3_3 --source zero --steps 800 \
 2>&1 | tee -a "$ROOT/train_split3_3_zero.log"
pids=()
for gpu in 0 1 2 3; do
 CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_csra_split_quality256.py sample \
  --root "$ROOT" --rank "$gpu" >> "$ROOT/split_quality256_gpu${gpu}.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed==0)) || { echo 'Split256 preview failed: inspect exact logs'; exit 1; }
CUDA_VISIBLE_DEVICES='' "$PY" 2d/attention_analysis/sample_csra_split_quality256.py audit --root "$ROOT"
echo 'Matched800-update zero-source control and fresh256-pair comparison complete; no auto extension.'
