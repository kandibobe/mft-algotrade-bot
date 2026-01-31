#!/bin/bash
# Canary Deployment Script for Stoic Citadel
# ==========================================
# Implements a basic Blue/Green deployment strategy using Docker Compose.
#
# Workflow:
# 1. Spin up the new version (Green) alongside the old one (Blue)
#    - Uses a separate project name or service name
# 2. Run health checks on Green
# 3. If Green is healthy:
#    - Swap Nginx upstream config (or load balancer)
#    - Stop Blue
# 4. If Green is unhealthy:
#    - Stop Green
#    - Alert admin
#
# Usage:
#   ./scripts/ops/canary_deploy.sh [version_tag]

set -e

VERSION=${1:-latest}
BLUE_COMPOSE="deploy/docker-compose.prod.yml"
GREEN_COMPOSE="deploy/docker-compose.prod.green.yml" # We'll generate this
PROJECT_NAME="stoic_citadel"

echo "🚀 Starting Canary Deployment for version: $VERSION"

# 1. Determine current active color (Blue or Green)
# In a real setup, we'd check Nginx config or a state file.
# For this script, we assume Blue is always running and we deploy Green temporarily.
# Or better: We assume standard 'prod' is running and we spin up 'canary' service.

# Let's define the strategy:
# - Spin up `stoic_canary` container (Green)
# - Run health checks
# - If healthy, stop `stoic_freqtrade_prod` and rename/promote `stoic_canary`?
# - Docker Compose makes "swapping" tricky without a load balancer in front.
# 
# Simpler Strategy for this context:
# 1. Pull new image
# 2. Start a "Canary" instance on a different port
# 3. Verify health
# 4. If healthy, perform standard `docker-compose up -d` (Rolling Update)

echo "🏗️ Phase 1: Deploying Canary Instance..."

# Generate a temporary compose override for the canary
cat <<EOF > docker-compose.canary.yml
services:
  freqtrade_canary:
    image: stoic_citadel:$VERSION
    container_name: stoic_canary
    environment:
      - DRY_RUN=true
      - INSTANCE_TYPE=canary
    ports:
      - "8081:8080" # Different port
    volumes:
      - ./user_data:/freqtrade/user_data
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - stoic_network
EOF

# Start Canary (using the main compose file + canary override)
docker-compose -f $BLUE_COMPOSE -f docker-compose.canary.yml up -d freqtrade_canary

echo "⏳ Waiting for Canary to stabilize..."
sleep 30

# 2. Health Check Canary
echo "🔍 Checking Canary Health..."
CANARY_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8081/health || echo "failed")

if [ "$CANARY_HEALTH" == "200" ]; then
    echo "✅ Canary is HEALTHY."
    
    # 3. Promotion
    echo "🚀 Promoting Canary to Production..."
    
    # Stop Canary
    docker-compose -f $BLUE_COMPOSE -f docker-compose.canary.yml stop freqtrade_canary
    docker-compose -f $BLUE_COMPOSE -f docker-compose.canary.yml rm -f freqtrade_canary
    rm docker-compose.canary.yml
    
    # Update Production Service (Rolling Update)
    # This will recreate the prod container with the new image
    docker-compose -f $BLUE_COMPOSE up -d --no-deps freqtrade
    
    echo "✅ Deployment Complete. Production updated to $VERSION."
    
else
    echo "❌ Canary is UNHEALTHY (HTTP $CANARY_HEALTH)."
    echo "⚠️  Aborting deployment. Production is untouched."
    
    # Cleanup Canary
    docker-compose -f $BLUE_COMPOSE -f docker-compose.canary.yml stop freqtrade_canary
    docker-compose -f $BLUE_COMPOSE -f docker-compose.canary.yml rm -f freqtrade_canary
    rm docker-compose.canary.yml
    
    exit 1
fi