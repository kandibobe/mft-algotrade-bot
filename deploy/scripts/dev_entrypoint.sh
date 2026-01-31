#!/bin/bash
set -e

# Default command if none provided
CMD="${@:-python manage.py trade}"

if command -v watchmedo &> /dev/null; then
    echo "🔍 Watchdog detected. Starting in HOT RELOAD mode."
    echo "📂 Watching: /freqtrade/user_data/src"
    
    # Watch src directory and restart the command on changes
    # --recursive: watch subdirectories
    # --pattern: watch python files
    # --signal: SIGTERM to stop process properly
    
    exec watchmedo auto-restart \
        --directory /freqtrade/user_data/src \
        --pattern "*.py" \
        --recursive \
        --signal SIGTERM \
        -- $CMD
else
    echo "⚠️ Watchdog NOT found. Starting in STANDARD mode."
    exec $CMD
fi