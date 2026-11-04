#!/usr/bin/env bash
# Four matched fresh800-update pilots, then paired small rollouts. No auto-best.
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
ROOT=/home/yuheng/rdd_experiments/csra_skip_layout_20260915
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
# Same transport workaround as all previous working four-GPU CSRA launchers
# on surfi2. Its P2P/IB path fails during NCCL initialization on this host.
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
mkdir -p "$ROOT"
exec 9>"$ROOT/layout_training.lock"
flock -n 9 || { echo 'This exact training queue already runs'; exit 1; }
layouts=(front0_6 front3_6 front6_6 split3_3)
while true; do
 ready=1
 for name in "${layouts[@]}"; do [[ -f "$ROOT/layouts/$name/metrics.json" ]] || ready=0; done
 ((ready==0)) || break
 if ! tmux has-session -t =csra_layer_probe_20260915 2>/dev/null; then
  echo 'Layer-probe prerequisite stopped without completion; inspect exact logs'; exit 1
 fi
 sleep 30
done
"$PY" -m unittest discover -s tests -p 'test_csra_layer_layouts.py'
"$PY" -m unittest discover -s tests -p 'test_csra_skip_schedule.py'
check_memory() {
 local m
 mapfile -t mem < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
 [[ ${#mem[@]} == 4 ]] || return 1
 for m in "${mem[@]}"; do ((m>=28672)) || return 1; done
}
for name in front0_6 split3_3; do
 until check_memory; do echo 'Waiting28GiB free on0-3 for4x64 smoke; other jobs untouched'; sleep 30; done
 CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
  2d/attention_analysis/train_csra_layout_pilot.py --root "$ROOT" --layout "$name" --steps 2 --smoke \
  2>&1 | tee -a "$ROOT/train_smoke_${name}.log"
done
for name in "${layouts[@]}"; do
 until check_memory; do echo 'Waiting28GiB free on0-3 for4x64 training; other jobs untouched'; sleep 30; done
 CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
  2d/attention_analysis/train_csra_layout_pilot.py --root "$ROOT" --layout "$name" --steps 800 \
  2>&1 | tee -a "$ROOT/train_${name}.log"
done
pids=()
for gpu in 0 1 2 3; do
 CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_csra_layout_pilot.py \
  --root "$ROOT" --rank "$gpu" >> "$ROOT/layout_quality_gpu${gpu}.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed==0)) || { echo 'Layout quality preview failed; inspect exact logs'; exit 1; }
echo 'Four800-update models and64 paired images/arm complete. Compare images before extending any arm.'
