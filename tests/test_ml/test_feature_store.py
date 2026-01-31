#!/usr/bin/env python3
"""
Tests for Feature Store implementation.
"""

from datetime import datetime
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.ml.feature_store import (
    RedisFeatureStore,
    get_feature_store,
    create_feature_store,
)


class TestRedisFeatureStore:
    """Test RedisFeatureStore (using mocks)."""

    @patch('src.ml.feature_store.redis.ConnectionPool.from_url')
    @patch('src.ml.feature_store.redis.Redis')
    def test_initialization(self, mock_redis, mock_pool):
        """Test that RedisFeatureStore initializes correctly."""
        store = RedisFeatureStore(redis_url="redis://localhost:6379", use_mock=True)
        assert store.redis_url == "redis://localhost:6379"
        assert store.use_mock is True
        assert store.ttl == 3600

    @patch('src.ml.feature_store.redis.Redis')
    @pytest.mark.asyncio
    async def test_get_online_features_mock(self, mock_redis):
        """Test get_online_features in mock mode."""
        # Force mock mode
        store = RedisFeatureStore(use_mock=True)
        
        symbol = "BTC/USDT"
        feature_list = ["feature_1", "feature_2"]
        
        # In mock mode, it should generate random features without calling Redis for data
        # (Though it might still try to check Redis if not careful, but the code says:
        # if self.use_mock: return self._generate_mock_features)
        
        features = await store.get_online_features(symbol, feature_list)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == 1
        assert "feature_1" in features.columns
        assert "feature_2" in features.columns
        assert features.iloc[0]["symbol"] == symbol

    @patch('src.ml.feature_store.redis.Redis')
    def test_set_features_batch(self, mock_redis):
        """Test batch storage of features."""
        store = RedisFeatureStore()
        # Mock pipeline
        mock_pipe = MagicMock()
        mock_redis.return_value.pipeline.return_value = mock_pipe
        
        df = pd.DataFrame([
            {"feature_1": 1.0, "feature_2": 2.0},
            {"feature_1": 3.0, "feature_2": 4.0}
        ], index=[datetime(2024, 1, 1), datetime(2024, 1, 2)])
        
        store.set_features_batch("BTC/USDT", df)
        
        # Check if setex was called for each row via pipeline
        assert mock_pipe.setex.call_count == 2
        mock_pipe.execute.assert_called_once()

    @patch('src.ml.feature_store.redis.Redis')
    def test_health_check(self, mock_redis):
        """Test health check."""
        store = RedisFeatureStore()
        
        # Success case
        mock_redis.return_value.ping.return_value = True
        mock_redis.return_value.dbsize.return_value = 10
        
        health = store.health_check()
        assert health["status"] == "healthy"
        assert health["cache_keys"] == 10
        
        # Failure case
        mock_redis.return_value.ping.side_effect = Exception("Connection error")
        health = store.health_check()
        assert health["status"] == "unhealthy"
        assert "error" in health


class TestFactoryFunctions:
    """Test the factory functions."""

    @patch('src.ml.feature_store.load_config')
    def test_get_feature_store(self, mock_load_config):
        """Test get_feature_store factory."""
        mock_cfg = MagicMock()
        mock_cfg.feature_store.redis_url = "redis://test:6379"
        mock_cfg.feature_store.enabled = True
        mock_load_config.return_value = mock_cfg
        
        store = get_feature_store()
        assert isinstance(store, RedisFeatureStore)
        assert store.redis_url == "redis://test:6379"
        assert store.use_mock is False

    @patch('src.ml.feature_store.get_feature_store')
    def test_create_feature_store_alias(self, mock_get):
        """Test legacy alias."""
        create_feature_store()
        mock_get.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
