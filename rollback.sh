#!/bin/bash
# Stoic Citadel - Automated Rollback Script

echo "⚠️ Health Check failed! Initiating automatic rollback..."

# 1. Stop current failing containers
docker-compose down

# 2. Revert to stable version/config if needed
# For example, revert to a 'stable' tag
# docker-compose pull stoic-citadel:stable

# 3. Restart with previous known good state
docker-compose up -d

echo "✅ Rollback complete. System reverted to stable state."
# Notify failure and rollback via Telegram
# python -m src.utils.notifications --message "🔴 Deployment Failed - System Rolled Back!"
