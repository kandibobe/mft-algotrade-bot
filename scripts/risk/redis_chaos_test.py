#!/usr/bin/env python3
"""
Redis Chaos Test
================

Tests the system's resilience to Redis failures by restarting the Redis container
while the bot is running in a simulated environment.

Usage:
    python scripts/risk/redis_chaos_test.py
"""

import asyncio
import logging
import time
import subprocess
import sys
from unittest.mock import MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
import os
sys.path.append(os.getcwd())

from src.ml.redis_client import RedisMLClient

async def chaos_test_redis():
    """
    Test scenario:
    1. Connect to Redis
    2. Start a data stream/interaction loop
    3. Kill Redis container
    4. Verify connection loss handling
    5. Restart Redis container
    6. Verify reconnection
    """
    
    logger.info("Starting Redis Chaos Test")
    
    # 1. Initialize Client
    redis_password = os.getenv("REDIS_PASSWORD", "redis_password_secure")
    redis_url = f"redis://:{redis_password}@localhost:6379"
    # Mask password in logs
    safe_url = redis_url.replace(redis_password, "***")
    
    client = RedisMLClient(url=redis_url)
    
    logger.info(f"Connecting to Redis at {safe_url}...")
    connected = await client.connect()
    
    if not connected:
        logger.error("Failed to initially connect to Redis. Is the container running?")
        logger.error("Try: docker-compose up -d redis")
        return False
        
    logger.info("Initial connection successful.")
    
    # 2. Start Interaction Loop
    running = True
    
    async def interaction_loop():
        counter = 0
        while running:
            try:
                # Try to write/read
                key = f"chaos_test:{counter}"
                await client.cache_set(key, {"value": counter}, ttl_seconds=10)
                result = await client.cache_get(key)
                
                if result and result.get("value") == counter:
                    logger.info(f"Loop {counter}: Redis operation successful")
                else:
                    logger.warning(f"Loop {counter}: Read mismatch or failure")
                
                # Check health
                health = await client.health_check()
                if health["status"] != "healthy":
                     logger.warning(f"Loop {counter}: Health check failed - {health.get('error')}")
                
            except Exception as e:
                logger.error(f"Loop {counter}: Operation failed: {e}")
                
            counter += 1
            await asyncio.sleep(1)

    task = asyncio.create_task(interaction_loop())
    
    # 3. Kill Redis Container
    logger.info("waiting 3 seconds before killing Redis...")
    await asyncio.sleep(3)
    
    logger.warning("CHAOS: Killing Redis container 'stoic_redis'...")
    try:
        subprocess.run(["docker", "kill", "stoic_redis"], check=True)
        logger.info("Redis container killed.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to kill Docker container: {e}")
        running = False
        await task
        return False

    # 4. Verify Handling (wait a few seconds of downtime)
    logger.info("Redis is down. Observing client behavior for 5 seconds...")
    await asyncio.sleep(5)
    
    # 5. Restart Redis Container
    logger.warning("CHAOS: Restarting Redis container 'stoic_redis'...")
    try:
        subprocess.run(["docker", "start", "stoic_redis"], check=True)
        logger.info("Redis container started.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start Docker container: {e}")
        running = False
        await task
        return False
        
    # 6. Verify Reconnection
    logger.info("Redis is up. Waiting for client reconnection (up to 15s)...")
    
    # Wait for a few successful loops
    await asyncio.sleep(10)
    
    running = False
    try:
        await task
    except asyncio.CancelledError:
        pass
        
    logger.info("Chaos test finished.")
    
    # Final health check
    health = await client.health_check()
    logger.info(f"Final Health Check: {health}")
    
    await client.disconnect()
    
    if health["status"] == "healthy":
        logger.info("SUCCESS: Client recovered from Redis failure.")
        return True
    else:
        logger.error("FAILURE: Client did not recover.")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(chaos_test_redis())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Test interrupted.")