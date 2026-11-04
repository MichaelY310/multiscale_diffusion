#!/bin/bash
# ============================================================
# 增量上传：全量日志 + 全量 checkpoint（50K 倍数 + latest）
# 
# 设计原则：
#   - Checkpoints: 自动扫描所有训练目录，只上传 50K 倍数 + 最新 ckpt
#   - Logs: 递归扫描整个 /mnt/bn/.../RDD/ 下所有 log.txt/timing_log.jsonl/*.log
#   - 本地日志: 所有 2d/、DiT/、根目录的 .log
#   - 使用 staging 目录 + 单次 hf upload 避免 rate limit (128 commits/hour)
#   - 可重复运行：HF 自动跳过内容相同的文件，不重复传输
#
# 自动检测新 checkpoint：
#   每次运行会扫描所有 checkpoints* 目录中的 .pt 文件，
#   上传 step 为 50K 倍数的 + 最新的那个。训练产出新 ckpt 后
#   再跑一次即可自动同步。
# ============================================================

set -e

# 网络配置
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export HTTPS_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=localhost,127.0.0.1
export HF_HUB_DISABLE_XET=1

REPO_ID="yang335/search_agent_hl_to_i18n_transport"
RDD_ROOT="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD"
LOCAL_ROOT="/home/tiger/multiscale_diffusion"

# Staging 目录：符号链接组织目录结构，最后一次性上传
STAGING="/tmp/hf_upload_staging_$$"
rm -rf "$STAGING"
mkdir -p "$STAGING"

echo "============================================"
echo " Incremental upload (batch mode)"
echo " Repo: $REPO_ID"
echo " Staging: $STAGING"
echo "============================================"

STAGED=0

# stage_file: 创建符号链接到 staging 目录（零拷贝，跨文件系统也秒完成）
stage_file() {
    local src="$1"
    local dst_rel="$2"  # 相对于 repo root 的目标路径
    local dst_full="$STAGING/$dst_rel"
    mkdir -p "$(dirname "$dst_full")"
    ln -sf "$(realpath "$src")" "$dst_full"
    STAGED=$((STAGED + 1))
}

# stage_ckpts_from_dir: 扫描一个 checkpoints 目录，上传 50K 倍数 + latest
stage_ckpts_from_dir() {
    local src_dir="$1"
    local dst_prefix="$2"
    [ -d "$src_dir" ] || return 0
    local latest=""
    local latest_num=0
    for f in "$src_dir"/*.pt; do
        [ -f "$f" ] || continue
        local step=$(basename "$f" .pt)
        local num=$((10#$step))
        if [ $num -gt $latest_num ]; then
            latest_num=$num
            latest="$f"
        fi
        if [ $((num % 50000)) -eq 0 ] && [ $num -gt 0 ]; then
            stage_file "$f" "${dst_prefix}/$(basename "$f")"
        fi
    done
    if [ -n "$latest" ]; then
        local latest_n=$((10#$(basename "$latest" .pt)))
        if [ $((latest_n % 50000)) -ne 0 ]; then
            stage_file "$latest" "${dst_prefix}/$(basename "$latest")"
        fi
    fi
}

# ═══════════════════════════════════════════════════════════════
# PART A: 远端训练数据（/mnt/bn/.../RDD/）
# ═══════════════════════════════════════════════════════════════

echo ""
echo "══════════════════════════════════════════════"
echo " PART A: Remote training data ($RDD_ROOT)"
echo "══════════════════════════════════════════════"

# ─── A1: 所有训练目录的 checkpoints ───
echo ""
echo "[A1] Scanning ALL checkpoint directories..."

# ImageNet_original 下的所有训练实验
for exp_dir in "$RDD_ROOT"/ImageNet_original/*/; do
    [ -d "$exp_dir" ] || continue
    exp_name=$(basename "$exp_dir")
    for sub_dir in "$exp_dir"/*/; do
        [ -d "$sub_dir" ] || continue
        # 扫描 checkpoints, checkpoints_8x8, checkpoints_16x16, checkpoints_32x32
        for ckpt_dir in "$sub_dir"/checkpoints*; do
            [ -d "$ckpt_dir" ] || continue
            ckpt_suffix=$(basename "$ckpt_dir")
            echo "  $exp_name/$(basename "$sub_dir")/$ckpt_suffix"
            stage_ckpts_from_dir "$ckpt_dir" "checkpoints/$exp_name/$ckpt_suffix"
        done
    done
done

# 顶层旧实验目录（RDD-DiT_ImageNet256_*、RDD-LDM_*）
for exp_dir in "$RDD_ROOT"/RDD-*/; do
    [ -d "$exp_dir" ] || continue
    exp_name=$(basename "$exp_dir")
    for sub_dir in "$exp_dir"/*/; do
        [ -d "$sub_dir" ] || continue
        for ckpt_dir in "$sub_dir"/checkpoints*; do
            [ -d "$ckpt_dir" ] || continue
            ckpt_suffix=$(basename "$ckpt_dir")
            echo "  $exp_name/$(basename "$sub_dir")/$ckpt_suffix"
            stage_ckpts_from_dir "$ckpt_dir" "checkpoints/$exp_name/$ckpt_suffix"
        done
    done
done

# ─── A2: 递归扫描所有远端日志 ───
echo ""
echo "[A2] Scanning ALL remote logs (log.txt, timing_log.jsonl, *.log)..."

# 使用 find 递归，保持相对路径结构
while IFS= read -r -d '' logfile; do
    rel_path="${logfile#$RDD_ROOT/}"
    stage_file "$logfile" "logs/remote/$rel_path"
done < <(find "$RDD_ROOT" -type f \( -name "log.txt" -o -name "timing_log.jsonl" -o -name "*.log" \) -print0 2>/dev/null)

echo "  Staged remote logs."

# ═══════════════════════════════════════════════════════════════
# PART B: 本地数据
# ═══════════════════════════════════════════════════════════════

echo ""
echo "══════════════════════════════════════════════"
echo " PART B: Local data ($LOCAL_ROOT)"
echo "══════════════════════════════════════════════"

# ─── B1: 2d/*.log ───
echo "[B1] Local training logs (2d/*.log)..."
for f in "$LOCAL_ROOT"/2d/*.log; do
    [ -f "$f" ] || continue
    stage_file "$f" "logs/local_training/$(basename "$f")"
done

# ─── B2: 2d/eval_results_unified/*.txt ───
echo "[B2] FID evaluation results..."
for f in "$LOCAL_ROOT"/2d/eval_results_unified/*.txt; do
    [ -f "$f" ] || continue
    stage_file "$f" "eval_results_unified/$(basename "$f")"
done

# ─── B3: 2d/exp/*.log ───
echo "[B3] Experiment logs (2d/exp/)..."
for f in "$LOCAL_ROOT"/2d/exp/*.log; do
    [ -f "$f" ] || continue
    stage_file "$f" "logs/exp/$(basename "$f")"
done

# ─── B4: 2d/logs_lowstep/*.log ───
echo "[B4] Low-step ablation logs..."
for f in "$LOCAL_ROOT"/2d/logs_lowstep/*.log; do
    [ -f "$f" ] || continue
    stage_file "$f" "logs/logs_lowstep/$(basename "$f")"
done

# ─── B5: 2d/logs/*.log ───
echo "[B5] Eval FID logs (2d/logs/)..."
for f in "$LOCAL_ROOT"/2d/logs/*.log; do
    [ -f "$f" ] || continue
    stage_file "$f" "logs/eval_fid/$(basename "$f")"
done

# ─── B6: 2d/speed_benchmark/*.txt ───
echo "[B6] Speed benchmark..."
for f in "$LOCAL_ROOT"/2d/speed_benchmark/*.txt; do
    [ -f "$f" ] || continue
    stage_file "$f" "logs/speed_benchmark/$(basename "$f")"
done

# ─── B7: DiT/*.log ───
echo "[B7] DiT eval logs..."
for f in "$LOCAL_ROOT"/DiT/*.log; do
    [ -f "$f" ] || continue
    stage_file "$f" "logs/DiT/$(basename "$f")"
done

# ─── B8: 根目录 .log 文件 ───
echo "[B8] Root-level logs..."
for f in "$LOCAL_ROOT"/*.log; do
    [ -f "$f" ] || continue
    stage_file "$f" "logs/root/$(basename "$f")"
done

# ═══════════════════════════════════════════════════════════════
# 一次性上传
# ═══════════════════════════════════════════════════════════════
echo ""
echo "============================================"
echo " Staged $STAGED files. Uploading in one batch..."
echo "============================================"

if [ $STAGED -eq 0 ]; then
    echo "Nothing to upload."
    rm -rf "$STAGING"
    exit 0
fi

# hf upload 会自动跳过远端已有且内容相同的文件（基于 SHA-256）
hf upload "$REPO_ID" "$STAGING" . --commit-message "Incremental sync: $STAGED files"

echo ""
echo "============================================"
echo " Done! Uploaded $STAGED files in single batch."
echo " Re-run this script anytime to sync new checkpoints/logs."
echo "============================================"

# 清理 staging 目录
rm -rf "$STAGING"
