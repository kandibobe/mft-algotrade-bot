#!/bin/bash
# Stoic Citadel - Automated Deployment Script with Health Check

echo "🚀 Starting deployment of Stoic Citadel..."

# 1. Pull latest images
docker-compose pull

# 2. Backup current state (if needed)
# ./scripts/maintenance/backup_db.sh

# 3. Start services in detached mode
docker-compose up -d

# 4. Wait for services to stabilize
echo "⏳ Waiting for services to start..."
sleep 15

# 5. Run Health Check
echo "🔍 Running Health Check..."
if python src/monitoring/health_check.py; then
    echo "✅ Deployment successful! System is healthy."
    # Notify success via Telegram (using existing utility)
    # python -m src.utils.notifications --message "Deployment Successful 🚀"
else
    echo "❌ Health Check failed! Triggering rollback..."
    ./rollback.sh
    exit 1
fi
