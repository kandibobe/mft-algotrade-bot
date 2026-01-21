#!/bin/bash
# Log Cleanup and Rotation Script for Stoic Citadel

LOG_DIR="user_data/logs"
MAX_LOG_SIZE_MB=50
MAX_BACKUPS=5

echo "🧹 Starting log maintenance in $LOG_DIR..."

# 1. Remove temporary test logs
rm -f "$LOG_DIR"/test_*.log

# 2. Rotate freqtrade.log if it's too large
if [ -f "$LOG_DIR/freqtrade.log" ]; then
    SIZE=$(du -m "$LOG_DIR/freqtrade.log" | cut -f1)
    if [ "$SIZE" -gt "$MAX_LOG_SIZE_MB" ]; then
        echo "🔄 freqtrade.log is $SIZE MB, rotating..."
        
        # Shift old backups
        for i in $(seq $((MAX_BACKUPS-1)) -1 1); do
            if [ -f "$LOG_DIR/freqtrade.log.$i" ]; then
                mv "$LOG_DIR/freqtrade.log.$i" "$LOG_DIR/freqtrade.log.$((i+1))"
            fi
        done
        
        # Backup current log
        mv "$LOG_DIR/freqtrade.log" "$LOG_DIR/freqtrade.log.1"
        touch "$LOG_DIR/freqtrade.log"
        echo "✅ Rotation complete."
    else
        echo "ℹ️ freqtrade.log size is $SIZE MB (under $MAX_LOG_SIZE_MB MB limit)."
    fi
fi

echo "✨ Log maintenance finished."