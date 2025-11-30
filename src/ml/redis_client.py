#!/usr/bin/env python3
"""
Redis Client for ML Inference
==============================

Simplified async Redis client wrapper for ML inference service.

Author: Stoic Citadel Team
License: MIT
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class RedisMLClient:
    """
    Simplified Redis client for ML inference communication.
    
    Handles:
    - Connection management with auto-reconnect
    - Pub/Sub for real-time updates
    - Queue operations for request/response
    - Health monitoring
    """
    
    def __init__(self, url: str = "redis://localhost:6379"):
        self.url = url
        self._redis = None
        self._pubsub = None
        self._connected = False
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 30.0
    
    async def connect(self) -> bool:
        """Establish connection to Redis."""
        try:
            import redis.asyncio as redis
            self._redis = redis.from_url(
                self.url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0
            )
            await self._redis.ping()
            self._connected = True
            self._reconnect_delay = 1.0  # Reset on successful connect
            logger.info(f"Connected to Redis at {self.url}")
            return True
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self._connected = False
            return False
    
    async def disconnect(self):
        """Close Redis connection."""
        if self._pubsub:
            await self._pubsub.close()
        if self._redis:
            await self._redis.close()
        self._connected = False
        logger.info("Disconnected from Redis")
    
    async def ensure_connected(self) -> bool:
        """Ensure connection is active, reconnect if needed."""
        if self._connected:
            try:
                await self._redis.ping()
                return True
            except:
                self._connected = False
        
        # Attempt reconnect with exponential backoff
        while not self._connected:
            if await self.connect():
                return True
            await asyncio.sleep(self._reconnect_delay)
            self._reconnect_delay = min(
                self._reconnect_delay * 2,
                self._max_reconnect_delay
            )
        return False
    
    # =========================================================================
    # Queue Operations (for request/response pattern)
    # =========================================================================
    
    async def push_request(self, queue: str, data: Dict[str, Any]) -> bool:
        """Push prediction request to queue."""
        try:
            await self.ensure_connected()
            await self._redis.lpush(queue, json.dumps(data))
            return True
        except Exception as e:
            logger.error(f"Failed to push request: {e}")
            return False
    
    async def pop_request(
        self,
        queues: List[str],
        timeout: float = 1.0
    ) -> Optional[tuple]:
        """Pop request from queue (blocking)."""
        try:
            await self.ensure_connected()
            result = await self._redis.brpop(queues, timeout=timeout)
            if result:
                queue_name, data = result
                return queue_name, json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Failed to pop request: {e}")
            return None
    
    async def push_result(self, queue: str, data: Dict[str, Any]) -> bool:
        """Push prediction result to queue."""
        try:
            await self.ensure_connected()
            await self._redis.lpush(queue, json.dumps(data))
            return True
        except Exception as e:
            logger.error(f"Failed to push result: {e}")
            return False
    
    # =========================================================================
    # Pub/Sub Operations (for real-time updates)
    # =========================================================================
    
    async def subscribe(self, channels: List[str]):
        """Subscribe to channels for real-time updates."""
        await self.ensure_connected()
        self._pubsub = self._redis.pubsub()
        await self._pubsub.subscribe(*channels)
        logger.info(f"Subscribed to channels: {channels}")
    
    async def publish(self, channel: str, message: Dict[str, Any]) -> int:
        """Publish message to channel."""
        try:
            await self.ensure_connected()
            return await self._redis.publish(channel, json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to publish: {e}")
            return 0
    
    async def listen(self):
        """Listen for messages on subscribed channels."""
        if not self._pubsub:
            raise RuntimeError("Not subscribed to any channels")
        
        async for message in self._pubsub.listen():
            if message["type"] == "message":
                yield {
                    "channel": message["channel"],
                    "data": json.loads(message["data"])
                }
    
    # =========================================================================
    # Cache Operations
    # =========================================================================
    
    async def cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached value."""
        try:
            await self.ensure_connected()
            value = await self._redis.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            return None
    
    async def cache_set(
        self,
        key: str,
        value: Dict[str, Any],
        ttl_seconds: int = 60
    ) -> bool:
        """Set cached value with TTL."""
        try:
            await self.ensure_connected()
            await self._redis.setex(key, ttl_seconds, json.dumps(value))
            return True
        except Exception as e:
            logger.error(f"Cache set failed: {e}")
            return False
    
    async def cache_delete(self, key: str) -> bool:
        """Delete cached value."""
        try:
            await self.ensure_connected()
            await self._redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete failed: {e}")
            return False
    
    # =========================================================================
    # Health & Stats
    # =========================================================================
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Redis health."""
        try:
            await self.ensure_connected()
            info = await self._redis.info()
            return {
                "status": "healthy",
                "connected": True,
                "redis_version": info.get("redis_version"),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "uptime_in_seconds": info.get("uptime_in_seconds")
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e)
            }
    
    async def get_queue_length(self, queue: str) -> int:
        """Get length of a queue."""
        try:
            await self.ensure_connected()
            return await self._redis.llen(queue)
        except:
            return -1
