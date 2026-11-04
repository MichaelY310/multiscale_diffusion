#!/bin/bash
# ============================================================
# 只上传日志/结果文件，不上传 checkpoint (.pt)
# 用于快速同步所有非 ckpt 文件
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

STAGING="/tmp/hf_upload_logs_$$"
rm -rf "$STAGING"
mkdir -p "$STAGING"

echo "============================================"
echo " Upload logs only (no checkpoints)"
echo " Repo: $REPO_ID"
echo " Staging: $STAGING"
echo "============================================"

STAGED=0

stage_file() {
    local src="$1"
    local dst_rel="$2"
    local dst_full="$STAGING/$dst_rel"
    mkdir -p "$(dirname "$dst_full")"
    ln -sf "$(realpath "$src")" "$dst_full"
    STAGED=$((STAGED + 1))
}

# ═══════════════════════════════════════════════════════════════
# 1. 远端所有日志（递归）
# ═══════════════════════════════════════════════════════════════
echo ""
echo "[1] Remote logs (recursive scan)..."
while IFS= read -r -d '' logfile; do
    rel_path="${logfile#$RDD_ROOT/}"
    stage_file "$logfile" "logs/remote/$rel_path"
done < <(find "$RDD_ROOT" -type f \( -name "*.txt" -o -name "timing_log.jsonl" -o -name "*.log" \) -not -path "*/samples*" -not -path "*checkpoints*" -print0 2>/dev/null)
echo "  Done scanning remote logs."

# ═══════════════════════════════════════════════════════════════
# 2. 本地日志
# ═══════════════════════════════════════════════════════════════
echo ""
echo "[2] Local logs..."

# 2d/*.log
for f in "$LOCAL_ROOT"/2d/*.log; do
    [ -f "$f" ] || continue
    stage_file "$f" "logs/local_training/$(basename "$f")"
done

# 2d/eval_results_unified/*.txt
for f in "$LOCAL_ROOT"/2d/eval_results_unified/*.txt; do
    [ -f "$f" ] || continue
    stage_file "$f" "eval_results_unified/$(basename "$f")"
done

# 2d/exp/eval_results_exp04/*.txt
for f in "$LOCAL_ROOT"/2d/exp/eval_results_exp04/*.txt; do
    [ -f "$f" ] || continue
    stage_file "$f" "eval_results_exp04/$(basename "$f")"
done

# 2d/exp/eval_results_dit500k/*.txt
for f in "$LOCAL_ROOT"/2d/exp/eval_results_dit500k/*.txt; do
    [ -f "$f" ] || continue
    stage_file "$f" "eval_results_dit500k/$(basename "$f")"
done

# 2d/exp/*.log
for f in "$LOCAL_ROOT"/2d/exp/*.log; do
    [ -f "$f" ] || continue
    stage_file "$f" "logs/exp/$(basename "$f")"
done

# 2d/logs_lowstep/*.log
for f in "$LOCAL_ROOT"/2d/logs_lowstep/*.log; do
    [ -f "$f" ] || continue
    stage_file "$f" "logs/logs_lowstep/$(basename "$f")"
done

# 2d/logs/*.log
for f in "$LOCAL_ROOT"/2d/logs/*.log; do
    [ -f "$f" ] || continue
    stage_file "$f" "logs/eval_fid/$(basename "$f")"
done

# 2d/speed_benchmark/*.txt + *.json
for f in "$LOCAL_ROOT"/2d/speed_benchmark/*.txt "$LOCAL_ROOT"/2d/speed_benchmark/*.json; do
    [ -f "$f" ] || continue
    stage_file "$f" "logs/speed_benchmark/$(basename "$f")"
done

# DiT/*.log
for f in "$LOCAL_ROOT"/DiT/*.log; do
    [ -f "$f" ] || continue
    stage_file "$f" "logs/DiT/$(basename "$f")"
done

# 根目录 *.log
for f in "$LOCAL_ROOT"/*.log; do
    [ -f "$f" ] || continue
    stage_file "$f" "logs/root/$(basename "$f")"
done

echo "  Done scanning local logs."

# ═══════════════════════════════════════════════════════════════
# 上传
# ═══════════════════════════════════════════════════════════════
echo ""
echo "============================================"
echo " Staged $STAGED files (logs only). Uploading..."
echo "============================================"

if [ $STAGED -eq 0 ]; then
    echo "Nothing to upload."
    rm -rf "$STAGING"
    exit 0
fi

hf upload "$REPO_ID" "$STAGING" . --commit-message "Sync all logs (no ckpt): $STAGED files"

echo ""
echo "============================================"
echo " Done! Uploaded $STAGED log files."
echo "============================================"

rm -rf "$STAGING"
