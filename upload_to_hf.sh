#!/bin/bash
# 上传精简版 checkpoint + 所有日志到 HuggingFace
# 仓库: yang335/search_agent_hl_to_i18n_transport (model 类型)
# 
# 使用前：
#   1. export HF_TOKEN=hf_xxxxxxx  (你的 write token)
#   2. 确保网络可达 huggingface.co (或设置 HF_ENDPOINT)

# 网络配置：设置代理访问 huggingface.co（上传必须走官方）
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export HTTPS_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=localhost,127.0.0.1
export HF_HUB_DISABLE_XET=1

# 不用 set -e，改为手动检查关键步骤，避免卡死
# set -e

REPO_ID="yang335/search_agent_hl_to_i18n_transport"
BASE="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original"
LOCAL_2D="/home/tiger/multiscale_diffusion/2d"
LOCAL_DIT="/home/tiger/multiscale_diffusion/DiT"
SAVED="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/saved"

# 创建仓库（如果不存在）
hf repos create "$REPO_ID" --type model 2>/dev/null || true

echo "============================================"
echo " Uploading to $REPO_ID"
echo "============================================"

# ══════════════════════════════════════════════
# 1. DiT baseline: 100K, 200K, 300K, 400K
# ══════════════════════════════════════════════
echo "[1/12] DiT-XL-2 checkpoints..."
for step in 0100000 0200000 0300000 0400000; do
  # 找到文件所在的编号目录
  f=$(ls $BASE/DiT-XL-2/*/checkpoints/${step}.pt 2>/dev/null | head -1)
  if [ -n "$f" ]; then
    hf upload "$REPO_ID" "$f" "checkpoints/DiT-XL-2/${step}.pt"
  else
    echo "  SKIP: DiT ${step} not found"
  fi
done

# ══════════════════════════════════════════════
# 2. RDD Unified XL: 100K, 200K, 300K, 400K, 750K
# ══════════════════════════════════════════════
echo "[2/12] RDD Unified XL checkpoints..."
for step in 0100000 0200000 0300000 0400000 0750000; do
  f="$BASE/RDD-Unified-XL-300-300-400/imagenet256-unified/checkpoints/${step}.pt"
  if [ -f "$f" ]; then
    hf upload "$REPO_ID" "$f" "checkpoints/RDD-Unified-XL/${step}.pt"
  else
    echo "  SKIP: UXL ${step} not found"
  fi
done

# ══════════════════════════════════════════════
# 3. RDD Unified L: 100K, 200K, 300K, 400K
# ══════════════════════════════════════════════
echo "[3/9] RDD Unified L checkpoints..."
for step in 0100000 0200000 0300000 0400000; do
  f="$BASE/RDD-Unified-L-300-300-400/imagenet256-unified/checkpoints/${step}.pt"
  if [ -f "$f" ]; then
    hf upload "$REPO_ID" "$f" "checkpoints/RDD-Unified-L/${step}.pt"
  else
    echo "  SKIP: UL ${step} not found"
  fi
done

# ══════════════════════════════════════════════
# 4. RDD Multistage: 只保留 400K (3 个分辨率)
# ══════════════════════════════════════════════
echo "[4/12] RDD Multistage 400K checkpoints..."
for res in 32x32 16x16 8x8; do
  f="$BASE/RDD-Multistage-300-300-400/imagenet256-multiscale/checkpoints_${res}/0400000.pt"
  if [ -f "$f" ]; then
    hf upload "$REPO_ID" "$f" "checkpoints/RDD-Multistage-300-300-400/checkpoints_${res}/0400000.pt"
  else
    echo "  SKIP: MS ${res} 400K not found"
  fi
done

# ══════════════════════════════════════════════
# 5. FM Multistage: 只保留 50K (3 个分辨率)
# ══════════════════════════════════════════════
echo "[5/9] FM Multistage 50K checkpoints..."
for res in 32x32 16x16 8x8; do
  f="$BASE/RDD-FM-Multistage-300-300-400/imagenet256-fm-multiscale/checkpoints_${res}/0050000.pt"
  if [ -f "$f" ]; then
    hf upload "$REPO_ID" "$f" "checkpoints/RDD-FM-Multistage/checkpoints_${res}/0050000.pt"
  else
    echo "  SKIP: FM-MS ${res} 50K not found"
  fi
done

# ══════════════════════════════════════════════
# 6. 训练日志 (ImageNet_original 下所有 log.txt / timing_log.jsonl)
# ══════════════════════════════════════════════
echo "[6/12] Training logs from ImageNet_original..."

# DiT logs
for d in $BASE/DiT-XL-2/*/; do
  name=$(basename "$d")
  [ -f "$d/log.txt" ] && hf upload "$REPO_ID" "$d/log.txt" "logs/DiT-XL-2/${name}/log.txt"
done
[ -f "$BASE/DiT-XL-2/train.log" ] && hf upload "$REPO_ID" "$BASE/DiT-XL-2/train.log" "logs/DiT-XL-2/train.log"

# RDD Multistage logs
for logf in log.txt timing_log.jsonl; do
  f="$BASE/RDD-Multistage-300-300-400/imagenet256-multiscale/$logf"
  [ -f "$f" ] && hf upload "$REPO_ID" "$f" "logs/RDD-Multistage-300-300-400/$logf"
done

# RDD Unified XL logs
for logf in log.txt timing_log.jsonl; do
  f="$BASE/RDD-Unified-XL-300-300-400/imagenet256-unified/$logf"
  [ -f "$f" ] && hf upload "$REPO_ID" "$f" "logs/RDD-Unified-XL/$logf"
done

# RDD Unified L logs
for logf in log.txt timing_log.jsonl; do
  f="$BASE/RDD-Unified-L-300-300-400/imagenet256-unified/$logf"
  [ -f "$f" ] && hf upload "$REPO_ID" "$f" "logs/RDD-Unified-L/$logf"
done

# FM Multistage logs
for logf in log.txt timing_log.jsonl; do
  f="$BASE/RDD-FM-Multistage-300-300-400/imagenet256-fm-multiscale/$logf"
  [ -f "$f" ] && hf upload "$REPO_ID" "$f" "logs/RDD-FM-Multistage/$logf"
done

# FM Unified XL v2 logs
for logf in log.txt timing_log.jsonl; do
  f="$BASE/RDD-FM-Unified-XL-v2/000-imagenet256-fm-unified/$logf"
  [ -f "$f" ] && hf upload "$REPO_ID" "$f" "logs/RDD-FM-Unified-XL-v2/$logf"
done

# ══════════════════════════════════════════════
# 7. 本地 FID 评估结果 (2d/*.txt)
# ══════════════════════════════════════════════
echo "[7/9] FID evaluation results..."
for f in $LOCAL_2D/*.txt; do
  [ -f "$f" ] && hf upload "$REPO_ID" "$f" "eval_results/$(basename $f)"
done

# ══════════════════════════════════════════════
# 8. Speed benchmark
# ══════════════════════════════════════════════
echo "[8/9] Speed benchmark..."
hf upload "$REPO_ID" "$LOCAL_2D/speed_benchmark/" "speed_benchmark/" --exclude "tmp_*"

# ══════════════════════════════════════════════
# 9. saved/ 目录下的所有日志
# ══════════════════════════════════════════════
echo "[9/12] Saved logs (old experiments)..."

# all_logs 下各个 eval_results 目录
for d in $SAVED/all_logs/eval_results_*; do
  [ -d "$d" ] && hf upload "$REPO_ID" "$d/" "saved/all_logs/$(basename $d)/"
done

# logs1 和 logs2
[ -d "$SAVED/all_logs/logs1" ] && hf upload "$REPO_ID" "$SAVED/all_logs/logs1/" "saved/all_logs/logs1/"
[ -d "$SAVED/all_logs/logs2" ] && hf upload "$REPO_ID" "$SAVED/all_logs/logs2/" "saved/all_logs/logs2/"

# old 目录的日志（不上传 .pt）
for d in $SAVED/old/*/; do
  name=$(basename "$d")
  # 递归找 log.txt 和 timing_log.jsonl
  for logf in $(find "$d" -name "log.txt" -o -name "timing_log.jsonl" 2>/dev/null); do
    relpath="${logf#$SAVED/}"
    hf upload "$REPO_ID" "$logf" "saved/$relpath"
  done
done

# DiT 本地日志
[ -d "$LOCAL_DIT/eval_results_1" ] && hf upload "$REPO_ID" "$LOCAL_DIT/eval_results_1/" "logs/DiT-eval/eval_results_1/"
[ -d "$LOCAL_DIT/eval_results_23" ] && hf upload "$REPO_ID" "$LOCAL_DIT/eval_results_23/" "logs/DiT-eval/eval_results_23/"
[ -d "$LOCAL_DIT/logs" ] && hf upload "$REPO_ID" "$LOCAL_DIT/logs/" "logs/DiT-eval/logs/"

# ══════════════════════════════════════════════
# 10. eval_results_unified/ (所有 FID 评估结果 txt)
# ══════════════════════════════════════════════
echo "[10/12] Unified FID evaluation results..."
[ -d "$LOCAL_2D/eval_results_unified" ] && hf upload "$REPO_ID" "$LOCAL_2D/eval_results_unified/" "eval_results_unified/"

# ══════════════════════════════════════════════
# 11. new_ablation 采样数据 (仅上传 FID npz 已删，上传 exp 配置)
# ══════════════════════════════════════════════
echo "[11/12] Experiment configs (exp/*.jsonl + exp/*.sh)..."
hf upload "$REPO_ID" "$LOCAL_2D/exp/" "exp_configs/" --exclude "tmp_*"

# ══════════════════════════════════════════════
# 12. DiT 500K checkpoint (如果已生成)
# ══════════════════════════════════════════════
echo "[12/12] DiT 500K checkpoint..."
f=$(ls $BASE/DiT-XL-2/*/checkpoints/0500000.pt 2>/dev/null | head -1)
if [ -n "$f" ]; then
  hf upload "$REPO_ID" "$f" "checkpoints/DiT-XL-2/0500000.pt"
else
  echo "  SKIP: DiT 500K not found yet"
fi

echo ""
echo "============================================"
echo " Upload complete!"
echo " Repo: https://huggingface.co/$REPO_ID"
echo "============================================"
