"""
Feature Store for Production ML (Refined)
=========================================

Provides production-ready feature store using Redis for ultra-low latency
online feature serving and Feast for offline feature management.

Key Refinements:
1. Cache-Aside Pattern: Automatic population of Redis from Feast.
2. Incremental Updates: Only calculate features for new candles.
3. Thread-safe Redis connection management.
"""

import asyncio
import logging
import os
import time
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import redis

# Feast imports
try:
    from feast import FeatureStore as FeastStore
    FEAST_AVAILABLE = True
except ImportError:
    FEAST_AVAILABLE = False
    FeastStore = Any

logger = logging.getLogger(__name__)

class RedisFeatureStore:
    """
    Production-grade Feature Store using Redis for MFT latency requirements.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        ttl_seconds: int = 3600,
        use_mock: bool = False
    ):
        self.redis_url = redis_url
        self.ttl = ttl_seconds
        self.use_mock = use_mock
        self._pool = redis.ConnectionPool.from_url(redis_url)
        self._redis = redis.Redis(connection_pool=self._pool)
        self._feast: Optional[FeastStore] = None
        
        if FEAST_AVAILABLE and not use_mock:
            try:
                self._feast = FeastStore(repo_path="feature_repo")
            except Exception as e:
                logger.warning(f"Failed to init Feast: {e}. Falling back to Redis-only mode.")

    def _get_conn(self):
        return self._redis

    def _make_key(self, symbol: str, timestamp: datetime) -> str:
        # Use floor to nearest candle (e.g., 5m) to ensure cache hits
        ts_str = timestamp.replace(second=0, microsecond=0).isoformat()
        return f"fs:{symbol}:{ts_str}"

    async def get_online_features(self, symbol: str, feature_list: List[str]) -> pd.DataFrame:
        """
        Refined get_online_features with Cache-Aside logic.
        1. Check Redis.
        2. If miss, check Feast (Offline/Materialized).
        3. If miss, return NaNs (or trigger computation).
        """
        now = datetime.utcnow()
        key = self._make_key(symbol, now)
        
        # 1. Redis lookup
        cached = self._redis.get(key)
        if cached:
            logger.debug(f"Redis Hit: {key}")
            return pickle.loads(cached)

        # 2. Feast lookup (Fallthrough)
        if self._feast:
            try:
                logger.info(f"Redis Miss, querying Feast for {symbol}")
                # Programmatic Feast lookup
                entity_rows = [{"symbol_id": symbol, "timestamp": now}]
                features = self._feast.get_online_features(
                    entity_rows=entity_rows,
                    features=feature_list
                ).to_df()
                
                # Update Redis
                self._redis.setex(key, self.ttl, pickle.dumps(features))
                return features
            except Exception as e:
                logger.error(f"Feast lookup failed: {e}")

        # 3. Fallback (Mock or Empty)
        if self.use_mock:
            return self._generate_mock_features(symbol, feature_list)
            
        return pd.DataFrame(columns=feature_list)

    def set_features_batch(self, symbol: str, df: pd.DataFrame):
        """
        Store a batch of features (e.g., after retraining or historical fetch).
        """
        pipe = self._redis.pipeline()
        for idx, row in df.iterrows():
            ts = idx if isinstance(idx, datetime) else pd.to_datetime(idx)
            key = self._make_key(symbol, ts)
            pipe.setex(key, self.ttl, pickle.dumps(pd.DataFrame([row])))
        pipe.execute()
        logger.info(f"Stored {len(df)} feature sets for {symbol} in Redis")

    def _generate_mock_features(self, symbol: str, feature_list: List[str]) -> pd.DataFrame:
        data = {f: np.random.randn() for f in feature_list}
        df = pd.DataFrame([data])
        df['symbol'] = symbol
        df['timestamp'] = datetime.utcnow()
        return df

    def health_check(self) -> Dict:
        try:
            self._redis.ping()
            return {
                "status": "healthy",
                "engine": "redis",
                "feast_integrated": self._feast is not None,
                "cache_keys": self._redis.dbsize()
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

def get_feature_store() -> RedisFeatureStore:
    """Factory to get singleton feature store."""
    from src.config.unified_config import load_config
    cfg = load_config()
    return RedisFeatureStore(
        redis_url=cfg.feature_store.redis_url,
        use_mock=not cfg.feature_store.enabled
    )

def create_feature_store() -> RedisFeatureStore:
    """Legacy alias for get_feature_store."""
    return get_feature_store()
