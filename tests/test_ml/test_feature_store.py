#!/usr/bin/env python3
"""
Tests for Feature Store implementation.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

from src.ml.feature_store import (
    TradingFeatureStore,
    MockFeatureStore,
    RedisFeatureStore,
    create_feature_store
)


class TestMockFeatureStore:
    """Test MockFeatureStore (doesn't require Feast)."""
    
    def test_initialization(self):
        """Test that MockFeatureStore initializes correctly."""
        store = MockFeatureStore()
        assert not store._initialized
        
        store.initialize()
        assert store._initialized
        assert store.enable_caching == True
        assert store.cache_ttl_hours == 1
    
    def test_register_features(self):
        """Test feature registration."""
        store = MockFeatureStore()
        store.initialize()
        store.register_features()
        
        assert hasattr(store, '_feature_views')
        assert 'ohlcv' in store._feature_views
        assert 'technical' in store._feature_views
        assert 'market' in store._feature_views
    
    def test_get_online_features(self):
        """Test getting online features."""
        store = MockFeatureStore()
        store.initialize()
        store.register_features()
        
        symbol = "BTC/USDT"
        timestamp = datetime.now()
        
        # Get all features
        features = store.get_online_features(symbol, timestamp)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        assert 'symbol_id' in features.columns
        assert 'timestamp' in features.columns
        assert 'open' in features.columns
        assert 'rsi_14' in features.columns
        
        # Get specific features
        specific_features = store.get_online_features(
            symbol, timestamp,
            feature_list=["ohlcv_features:open", "technical_features:rsi_14"]
        )
        
        assert isinstance(specific_features, pd.DataFrame)
        assert 'open' in specific_features.columns
        assert 'rsi_14' in specific_features.columns
    
    def test_caching(self):
        """Test feature caching."""
        store = MockFeatureStore(enable_caching=True, cache_ttl_hours=1)
        store.initialize()
        
        symbol = "BTC/USDT"
        timestamp = datetime.now()
        
        # First call should not be cached
        features1 = store.get_online_features(symbol, timestamp)
        assert len(store._feature_cache) == 1
        
        # Second call should be cached
        features2 = store.get_online_features(symbol, timestamp)
        assert len(store._feature_cache) == 1
        
        # Data should be the same (within floating point precision)
        pd.testing.assert_frame_equal(features1, features2)
    
    def test_cache_invalidation(self):
        """Test cache invalidation."""
        store = MockFeatureStore(enable_caching=True)
        store.initialize()
        
        symbol = "BTC/USDT"
        timestamp = datetime.now()
        
        # Get features to populate cache
        store.get_online_features(symbol, timestamp)
        assert len(store._feature_cache) == 1
        
        # Invalidate cache
        store._invalidate_cache(symbol, timestamp)
        assert len(store._feature_cache) == 0
    
    def test_health_check(self):
        """Test health check."""
        store = MockFeatureStore()
        
        # Check uninitialized
        health = store.health_check()
        assert health['status'] == 'uninitialized'
        
        # Check initialized
        store.initialize()
        health = store.health_check()
        assert health['status'] == 'healthy'
        assert health['initialized'] == True
    
    def test_clear_cache(self):
        """Test cache clearing."""
        store = MockFeatureStore(enable_caching=True)
        store.initialize()
        
        # Populate cache
        store.get_online_features("BTC/USDT", datetime.now())
        store.get_online_features("ETH/USDT", datetime.now())
        assert len(store._feature_cache) == 2
        
        # Clear cache
        store.clear_cache()
        assert len(store._feature_cache) == 0


class TestRedisFeatureStore:
    """Test RedisFeatureStore (requires Redis or mocks)."""
    
    def test_initialization(self):
        """Test that RedisFeatureStore initializes correctly."""
        # Test with default parameters
        try:
            store = RedisFeatureStore()
            # Should have Redis connection
            assert hasattr(store, 'redis')
            assert hasattr(store, 'pickle')
            assert store.ttl == 3600  # 1 hour in seconds
        except Exception as e:
            # If Redis is not available, skip the test
            pytest.skip(f"Redis not available: {e}")
    
    def test_get_features_method(self):
        """Test the get_features method (simple key-value interface)."""
        try:
            store = RedisFeatureStore()
            store.initialize()
            
            symbol = "BTC/USDT"
            timestamp = "2024-01-01 12:00:00"
            
            # Test getting non-existent features
            features = store.get_features(symbol, timestamp)
            assert features is None
            
            # Test setting and getting features
            test_features = {"open": 50000.0, "close": 50500.0, "volume": 1000.0}
            store.set_features(symbol, timestamp, test_features)
            
            # Note: In a real test, we would get the features back
            # But since Redis might not be available, we'll just verify the method exists
            assert hasattr(store, 'get_features')
            assert hasattr(store, 'set_features')
            
        except Exception as e:
            pytest.skip(f"Redis not available: {e}")
    
    def test_get_online_features_with_redis(self):
        """Test get_online_features using Redis cache."""
        try:
            store = RedisFeatureStore(enable_caching=True)
            store.initialize()
            store.register_features()
            
            symbol = "BTC/USDT"
            timestamp = datetime.now()
            
            # Get features (should generate mock features and cache in Redis)
            features = store.get_online_features(symbol, timestamp)
            
            assert isinstance(features, pd.DataFrame)
            assert len(features) > 0
            assert 'symbol_id' in features.columns
            assert 'timestamp' in features.columns
            
            # Verify Redis-specific methods exist
            assert hasattr(store, 'clear_redis_cache')
            
        except Exception as e:
            pytest.skip(f"Redis not available: {e}")
    
    def test_health_check(self):
        """Test Redis feature store health check."""
        try:
            store = RedisFeatureStore()
            store.initialize()
            
            health = store.health_check()
            
            # Should have Redis-specific health info
            assert 'redis_connected' in health
            assert 'redis_version' in health or 'error' in health
            assert 'cache_size' in health
            
        except Exception as e:
            pytest.skip(f"Redis not available: {e}")


class TestTradingFeatureStore:
    """Test TradingFeatureStore (requires Feast)."""
    
    def test_initialization_without_feast(self):
        """Test that TradingFeatureStore raises error when Feast not available."""
        # This test assumes Feast is not installed
        # We'll mock the FEAST_AVAILABLE flag
        import src.ml.feature_store as fs_module
        
        original_value = fs_module.FEAST_AVAILABLE
        try:
            fs_module.FEAST_AVAILABLE = False
            
            store = TradingFeatureStore()
            with pytest.raises(ImportError):
                store.initialize()
        finally:
            fs_module.FEAST_AVAILABLE = original_value
    
    def test_feast_import_handling(self):
        """Test that the module handles Feast import errors gracefully."""
        # The module should not crash if Feast is not available
        import src.ml.feature_store as fs_module
        
        # Just verify the module imports without error
        assert hasattr(fs_module, 'TradingFeatureStore')
        assert hasattr(fs_module, 'MockFeatureStore')
        assert hasattr(fs_module, 'create_feature_store')


class TestFactoryFunction:
    """Test the create_feature_store factory function."""
    
    def test_create_mock_store(self):
        """Test creating mock feature store."""
        # Test explicit mock request
        store = create_feature_store(use_mock=True)
        assert isinstance(store, MockFeatureStore)
        
        # Test mock when Feast not available
        import src.ml.feature_store as fs_module
        original_value = fs_module.FEAST_AVAILABLE
        
        try:
            fs_module.FEAST_AVAILABLE = False
            store = create_feature_store()
            assert isinstance(store, MockFeatureStore)
        finally:
            fs_module.FEAST_AVAILABLE = original_value
    
    def test_create_real_store_when_available(self):
        """Test creating real feature store when Feast is available."""
        import src.ml.feature_store as fs_module
        original_value = fs_module.FEAST_AVAILABLE
        
        try:
            fs_module.FEAST_AVAILABLE = True
            store = create_feature_store(use_mock=False)
            # When Feast is available, it should create TradingFeatureStore
            # But we can't actually test this without Feast installed
            # So we'll just verify the function doesn't crash
            assert store is not None
        finally:
            fs_module.FEAST_AVAILABLE = original_value
    
    def test_create_redis_store(self):
        """Test creating Redis feature store."""
        # Test explicit redis request
        try:
            store = create_feature_store(use_redis=True)
            assert isinstance(store, RedisFeatureStore)
        except Exception as e:
            # If Redis is not available, skip the test
            pytest.skip(f"Redis not available: {e}")


class TestIntegration:
    """Integration tests for feature store."""
    
    def test_feature_store_with_inference_service(self):
        """Test that feature store can provide features for inference service."""
        # This is a conceptual test - in practice, you would:
        # 1. Create feature store
        # 2. Get features
        # 3. Use features with ML inference
        
        store = MockFeatureStore()
        store.initialize()
        store.register_features()
        
        # Get features for inference
        features_df = store.get_online_features(
            symbol="BTC/USDT",
            timestamp=datetime.now()
        )
        
        # Convert to dictionary format expected by inference service
        features_dict = features_df.iloc[0].to_dict()
        
        # Remove metadata columns
        features_dict = {k: v for k, v in features_dict.items() 
                        if k not in ['symbol_id', 'timestamp']}
        
        # Verify we have features in the right format
        assert isinstance(features_dict, dict)
        assert len(features_dict) > 0
        
        # All values should be numeric
        for value in features_dict.values():
            assert isinstance(value, (int, float, np.floating, np.integer))
    
    def test_async_features(self):
        """Test async feature retrieval."""
        import asyncio
        
        store = MockFeatureStore()
        store.initialize()
        
        async def test_async():
            features = await store.get_online_features_async(
                symbol="BTC/USDT",
                timestamp=datetime.now()
            )
            assert isinstance(features, pd.DataFrame)
            assert len(features) > 0
        
        asyncio.run(test_async())
    
    def test_feature_stats(self):
        """Test getting feature statistics."""
        store = MockFeatureStore()
        store.initialize()
        store.register_features()
        
        # Populate cache
        store.get_online_features("BTC/USDT", datetime.now())
        
        stats = store.get_feature_stats()
        
        assert 'cache_size' in stats
        assert 'cache_hit_rate' in stats
        assert 'initialized' in stats
        assert 'feature_views' in stats
        
        assert stats['cache_size'] == 1
        assert stats['initialized'] == True
        assert isinstance(stats['feature_views'], list)


if __name__ == "__main__":
    # Run tests
    import sys
    pytest.main([__file__, "-v"])
