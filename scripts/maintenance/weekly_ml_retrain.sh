#!/bin/bash
# Stoic Citadel - Weekly ML Retraining Script
# This script is intended to be run via cron weekly to keep models fresh.

set -e

# Project directory
PROJECT_DIR="c:/mft-algotrade-bot"
cd "$PROJECT_DIR"

# Activate virtual environment (if exists)
if [ -d ".venv" ]; then
    source .venv/bin/activate || source .venv/Scripts/activate
fi

echo "Starting weekly ML model retraining: $(date)"

# Run full ML pipeline retraining
# This will retrain models for all whitelisted pairs and save them to production
python -m src.ml.training.train --all --production

echo "Model retraining completed successfully: $(date)"
