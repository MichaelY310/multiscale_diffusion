#!/bin/bash
# Wait until ImageNet unzip is complete by monitoring JPEG file count.
# Exits 0 when count stabilizes, exits 1 on timeout.

TARGET_DIR="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/data/imagenet/ILSVRC/Data/CLS-LOC/train"
EXPECTED=1281167
INTERVAL=30

echo "[$(date '+%H:%M:%S')] Monitoring: $TARGET_DIR"
echo "[$(date '+%H:%M:%S')] Expected: $EXPECTED JPEGs"
echo ""

prev=0
stable_count=0

while true; do
    curr=$(find "$TARGET_DIR" -name "*.JPEG" 2>/dev/null | wc -l)
    pct=$(awk "BEGIN {printf \"%.1f\", $curr / $EXPECTED * 100}")

    echo "[$(date '+%H:%M:%S')] ${curr} / ${EXPECTED} (${pct}%)"

    if [ "$curr" -ge "$EXPECTED" ]; then
        echo ""
        echo "[$(date '+%H:%M:%S')] SUCCESS: All $curr JPEGs extracted."
        exit 0
    fi

    if [ "$curr" -eq "$prev" ]; then
        stable_count=$((stable_count + 1))
        if [ $stable_count -ge 3 ]; then
            echo "[$(date '+%H:%M:%S')] Count stable at $curr (${pct}%) but below expected. Still waiting..."
            stable_count=0
        fi
    else
        stable_count=0
    fi

    prev=$curr
    sleep $INTERVAL
done
