#!/usr/bin/env bash
# One bounded matched extension. Preserve every800 checkpoint and result.
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
ROOT=/home/yuheng/rdd_experiments/csra_skip_layout_20260915
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
exec 9>"$ROOT/continuation1600.lock"
flock -n 9 || { echo 'Exact continuation already runs'; exit 1; }
CUDA_VISIBLE_DEVICES='' "$PY" 2d/attention_analysis/summarize_csra_split_video_cost.py "$ROOT/split_video_cost"
CUDA_VISIBLE_DEVICES='' "$PY" -c 'import json,sys; m=json.load(open(sys.argv[1])); rows=[r for r in m["rows"] if r["case"].startswith("1280") and r["variant"]=="split3_3"]; assert len(rows)==2 and all(r["paired_reduction_ci95"]["denoise"]["lower"]>10 for r in rows), "Video denoise cost gate failed"' "$ROOT/split_video_cost/summary_and_audit.json"
check_memory() {
 local m
 mapfile -t mem < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
 [[ ${#mem[@]} == 4 ]] || return 1
 for m in "${mem[@]}"; do ((m>=28672)) || return 1; done
}
until check_memory; do echo 'Waiting28GiB free on0-3;4-7 reserved'; sleep 30; done
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
 2d/attention_analysis/continue_csra_layout_to1600.py --root "$ROOT" --arm split3_3_aligned --smoke \
 2>&1 | tee -a "$ROOT/continuation1600_smoke.log"
for arm in split3_3_aligned front0_6_aligned split3_3_zero; do
 CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
  2d/attention_analysis/continue_csra_layout_to1600.py --root "$ROOT" --arm "$arm" \
  2>&1 | tee -a "$ROOT/continuation1600_${arm}.log"
done
pids=()
for gpu in 0 1 2 3; do
 CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_csra_split1600_quality256.py sample \
  --root "$ROOT" --rank "$gpu" >> "$ROOT/split1600_quality256_gpu${gpu}.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed==0)) || { echo 'Matched1600 sampling failed; inspect exact logs'; exit 1; }
CUDA_VISIBLE_DEVICES='' "$PY" 2d/attention_analysis/sample_csra_split1600_quality256.py audit --root "$ROOT"
echo 'Matched1600 quality complete; no auto extension beyond1600.'
