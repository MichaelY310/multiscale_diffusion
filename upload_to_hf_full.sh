#!/bin/bash
# 完整上传：所有 50K 倍数 checkpoint + 全部日志
# 仓库: yang335/search_agent_hl_to_i18n_transport
# 
# 这个脚本上传精简版之外的所有 50K 倍数 ckpt
# 精简版已上传的不会重复上传（hf upload 会跳过已存在的文件）

# 网络配置
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export HTTPS_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=localhost,127.0.0.1
export HF_HUB_DISABLE_XET=1

REPO_ID="yang335/search_agent_hl_to_i18n_transport"
BASE="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original"
LOCAL_2D="/home/tiger/multiscale_diffusion/2d"
LOCAL_DIT="/home/tiger/multiscale_diffusion/DiT"
SAVED="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/saved"

echo "============================================"
echo " Full upload to $REPO_ID"
echo "============================================"

# ══════════════════════════════════════════════
# 1. DiT-XL-2: 全部 50K 倍数 (分散在多个编号目录)
# ══════════════════════════════════════════════
echo "[1/10] DiT-XL-2 ALL 50K-multiple checkpoints..."
for f in $BASE/DiT-XL-2/*/checkpoints/*.pt; do
  [ -f "$f" ] || continue
  step=$(basename "$f" .pt)
  num=$((10#$step))
  if [ $((num % 50000)) -eq 0 ]; then
    echo "  Uploading DiT ${step}..."
    hf upload "$REPO_ID" "$f" "checkpoints/DiT-XL-2/${step}.pt"
  fi
done

# ══════════════════════════════════════════════
# 2. RDD Unified XL: 全部 50K 倍数 + 最新 ckpt
# ══════════════════════════════════════════════
echo "[2/10] RDD Unified XL ALL 50K-multiple checkpoints..."
for f in $BASE/RDD-Unified-XL-300-300-400/imagenet256-unified/checkpoints/*.pt; do
  [ -f "$f" ] || continue
  step=$(basename "$f" .pt)
  num=$((10#$step))
  if [ $((num % 50000)) -eq 0 ]; then
    echo "  Uploading UXL ${step}..."
    hf upload "$REPO_ID" "$f" "checkpoints/RDD-Unified-XL/${step}.pt"
  fi
done
# 最新 ckpt (非 50K 倍数)
latest=$(ls $BASE/RDD-Unified-XL-300-300-400/imagenet256-unified/checkpoints/*.pt 2>/dev/null | sort | tail -1)
if [ -n "$latest" ]; then
  step=$(basename "$latest" .pt)
  echo "  Uploading UXL latest: ${step}..."
  hf upload "$REPO_ID" "$latest" "checkpoints/RDD-Unified-XL/${step}.pt"
fi

# ══════════════════════════════════════════════
# 3. RDD Unified L: 全部 50K 倍数 + 最新 ckpt
# ══════════════════════════════════════════════
echo "[3/10] RDD Unified L ALL 50K-multiple checkpoints..."
for f in $BASE/RDD-Unified-L-300-300-400/imagenet256-unified/checkpoints/*.pt; do
  [ -f "$f" ] || continue
  step=$(basename "$f" .pt)
  num=$((10#$step))
  if [ $((num % 50000)) -eq 0 ]; then
    echo "  Uploading UL ${step}..."
    hf upload "$REPO_ID" "$f" "checkpoints/RDD-Unified-L/${step}.pt"
  fi
done
latest=$(ls $BASE/RDD-Unified-L-300-300-400/imagenet256-unified/checkpoints/*.pt 2>/dev/null | sort | tail -1)
if [ -n "$latest" ]; then
  step=$(basename "$latest" .pt)
  echo "  Uploading UL latest: ${step}..."
  hf upload "$REPO_ID" "$latest" "checkpoints/RDD-Unified-L/${step}.pt"
fi

# ══════════════════════════════════════════════
# 4. RDD Multistage: 全部 50K 倍数 (3 个分辨率)
# ══════════════════════════════════════════════
echo "[4/10] RDD Multistage ALL 50K-multiple checkpoints..."
for res in 32x32 16x16 8x8; do
  for f in $BASE/RDD-Multistage-300-300-400/imagenet256-multiscale/checkpoints_${res}/*.pt; do
    [ -f "$f" ] || continue
    step=$(basename "$f" .pt)
    num=$((10#$step))
    if [ $((num % 50000)) -eq 0 ]; then
      echo "  Uploading MS ${res} ${step}..."
      hf upload "$REPO_ID" "$f" "checkpoints/RDD-Multistage-300-300-400/checkpoints_${res}/${step}.pt"
    fi
  done
done

# ══════════════════════════════════════════════
# 5. FM Multistage: 全部 (3 个分辨率)
# ══════════════════════════════════════════════
echo "[5/10] FM Multistage ALL checkpoints..."
for res in 32x32 16x16 8x8; do
  for f in $BASE/RDD-FM-Multistage-300-300-400/imagenet256-fm-multiscale/checkpoints_${res}/*.pt; do
    [ -f "$f" ] || continue
    step=$(basename "$f" .pt)
    num=$((10#$step))
    if [ $((num % 50000)) -eq 0 ]; then
      echo "  Uploading FM-MS ${res} ${step}..."
      hf upload "$REPO_ID" "$f" "checkpoints/RDD-FM-Multistage/checkpoints_${res}/${step}.pt"
    fi
  done
  # 最新
  latest=$(ls $BASE/RDD-FM-Multistage-300-300-400/imagenet256-fm-multiscale/checkpoints_${res}/*.pt 2>/dev/null | sort | tail -1)
  if [ -n "$latest" ]; then
    step=$(basename "$latest" .pt)
    echo "  Uploading FM-MS ${res} latest: ${step}..."
    hf upload "$REPO_ID" "$latest" "checkpoints/RDD-FM-Multistage/checkpoints_${res}/${step}.pt"
  fi
done

# ══════════════════════════════════════════════
# 6. FM Unified XL v2: 全部
# ══════════════════════════════════════════════
echo "[6/10] FM Unified XL v2 ALL checkpoints..."
for f in $BASE/RDD-FM-Unified-XL-v2/000-imagenet256-fm-unified/checkpoints/*.pt; do
  [ -f "$f" ] || continue
  step=$(basename "$f" .pt)
  echo "  Uploading FM-UXL ${step}..."
  hf upload "$REPO_ID" "$f" "checkpoints/RDD-FM-Unified-XL-v2/${step}.pt"
done

# ══════════════════════════════════════════════
# 7. 训练日志 (ImageNet_original)
# ══════════════════════════════════════════════
echo "[7/10] Training logs..."

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

# Multistage 100-300-600 logs (正在训练中的)
for logf in log.txt timing_log.jsonl; do
  f="$BASE/RDD-Multistage-100-300-600/imagenet256-multiscale/$logf"
  [ -f "$f" ] && hf upload "$REPO_ID" "$f" "logs/RDD-Multistage-100-300-600/$logf"
done

# ══════════════════════════════════════════════
# 8. 本地 FID 评估结果 (2d/*.txt)
# ══════════════════════════════════════════════
echo "[8/10] FID evaluation results..."
for f in $LOCAL_2D/*.txt; do
  [ -f "$f" ] && hf upload "$REPO_ID" "$f" "eval_results/$(basename $f)"
done

# ══════════════════════════════════════════════
# 9. Speed benchmark
# ══════════════════════════════════════════════
echo "[9/10] Speed benchmark..."
hf upload "$REPO_ID" "$LOCAL_2D/speed_benchmark/" "speed_benchmark/" --exclude "tmp_*"

# ══════════════════════════════════════════════
# 10. saved/ 目录下的所有日志 + DiT 本地日志
# ══════════════════════════════════════════════
echo "[10/10] Saved logs (old experiments)..."

# all_logs 下各个 eval_results 目录
for d in $SAVED/all_logs/eval_results_*; do
  [ -d "$d" ] && hf upload "$REPO_ID" "$d/" "saved/all_logs/$(basename $d)/"
done

# logs1 和 logs2
[ -d "$SAVED/all_logs/logs1" ] && hf upload "$REPO_ID" "$SAVED/all_logs/logs1/" "saved/all_logs/logs1/"
[ -d "$SAVED/all_logs/logs2" ] && hf upload "$REPO_ID" "$SAVED/all_logs/logs2/" "saved/all_logs/logs2/"

# old 目录的日志（不上传 .pt）
for d in $SAVED/old/*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  for logf in $(find "$d" -name "log.txt" -o -name "timing_log.jsonl" 2>/dev/null); do
    relpath="${logf#$SAVED/}"
    hf upload "$REPO_ID" "$logf" "saved/$relpath"
  done
done

# 旧 HF 数据集实验的日志（ckpt不上传）
OLD_RDD="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD"
for exp in RDD-DiT_ImageNet256_300_300_400 RDD-DiT_ImageNet256_Unified_300_300_400 RDD-DiT_ImageNet256_Unified_300_300_400_L2 RDD-DiT_ImageNet512_Unified_100_200_300_400 RDD-DiT_ImageNet256_100_300_600_resumed; do
  [ -d "$OLD_RDD/$exp" ] || continue
  for logf in $(find "$OLD_RDD/$exp" -name "log.txt" -o -name "timing_log.jsonl" 2>/dev/null); do
    relpath="${logf#$OLD_RDD/}"
    hf upload "$REPO_ID" "$logf" "logs/old_hf_dataset/$relpath"
  done
done

# DiT 本地日志
[ -d "$LOCAL_DIT/eval_results_1" ] && hf upload "$REPO_ID" "$LOCAL_DIT/eval_results_1/" "logs/DiT-eval/eval_results_1/"
[ -d "$LOCAL_DIT/eval_results_23" ] && hf upload "$REPO_ID" "$LOCAL_DIT/eval_results_23/" "logs/DiT-eval/eval_results_23/"
[ -d "$LOCAL_DIT/logs" ] && hf upload "$REPO_ID" "$LOCAL_DIT/logs/" "logs/DiT-eval/logs/"
[ -f "$LOCAL_DIT/train_imagenet512.log" ] && hf upload "$REPO_ID" "$LOCAL_DIT/train_imagenet512.log" "logs/DiT-eval/train_imagenet512.log"

# 本地 exp 目录（实验配置）
hf upload "$REPO_ID" "$LOCAL_2D/exp/" "exp/" --exclude "*.log"

echo ""
echo "============================================"
echo " Full upload complete!"
echo " Repo: https://huggingface.co/$REPO_ID"
echo "============================================"
