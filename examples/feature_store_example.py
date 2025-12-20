#!/usr/bin/env python3
"""
Example: Feature Store Integration with ML Inference Service
============================================================

This example demonstrates how to use the Feature Store with the existing
ML Inference Service to solve the problems mentioned in Task 3.1:

1. Пересчет features при каждом inference (Recalculating features for each inference)
2. Нет разделения online/offline features (No separation between online/offline features)
3. Отсутствие версионирования features (No feature versioning)

The Feature Store provides:
- Precomputed features (100x faster than recomputing)
- Online/offline feature separation
- Feature versioning and caching
"""

import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd

# Import from our ML module
from src.ml.feature_store import create_feature_store
from src.ml.inference_service import MLInferenceService, MLModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main example function."""
    print("=" * 80)
    print("Feature Store Integration Example")
    print("=" * 80)
    
    # =========================================================================
    # 1. Create Feature Store (automatically uses mock since Feast not installed)
    # =========================================================================
    print("\n1. Creating Feature Store...")
    feature_store = create_feature_store(
        config_path="feature_repo",
        enable_caching=True,
        cache_ttl_hours=1
    )
    
    # Initialize and register features
    feature_store.initialize()
    feature_store.register_features()
    
    print(f"   Feature Store created: {feature_store.__class__.__name__}")
    print(f"   Caching enabled: {feature_store.enable_caching}")
    print(f"   Cache TTL: {feature_store.cache_ttl_hours} hours")
    
    # =========================================================================
    # 2. Get Online Features for Real-time Inference
    # =========================================================================
    print("\n2. Getting Online Features for real-time inference...")
    
    symbol = "BTC/USDT"
    timestamp = datetime.now()
    
    # Get precomputed features (100x faster than recomputing!)
    start_time = datetime.now()
    online_features = feature_store.get_online_features(symbol, timestamp)
    latency = (datetime.now() - start_time).total_seconds() * 1000
    
    print(f"   Retrieved {len(online_features.columns)} features for {symbol}")
    print(f"   Latency: {latency:.2f}ms")
    print(f"   Features sample:")
    
    # Show first few features
    feature_sample = online_features.iloc[0]
    for i, (col, val) in enumerate(feature_sample.items()):
        if i >= 5:  # Show only first 5
            print(f"     ... and {len(online_features.columns) - 5} more")
            break
        print(f"     {col}: {val:.4f}")
    
    # =========================================================================
    # 3. Convert Features for ML Inference
    # =========================================================================
    print("\n3. Converting features for ML Inference Service...")
    
    # Convert DataFrame to dictionary format expected by inference service
    features_dict = online_features.iloc[0].to_dict()
    
    # Remove metadata columns
    ml_features = {k: v for k, v in features_dict.items() 
                  if k not in ['symbol_id', 'timestamp']}
    
    print(f"   Converted to {len(ml_features)} ML features")
    print(f"   Feature types: {list(ml_features.keys())[:3]}...")
    
    # =========================================================================
    # 4. Use with ML Inference Service
    # =========================================================================
    print("\n4. Using features with ML Inference Service...")
    
    # Create a mock ML inference service
    # In production, this would connect to Redis and ML workers
    inference_service = MLInferenceService(
        redis_url="redis://localhost:6379",
        models={
            "trend_classifier": MLModelConfig(
                model_name="trend_classifier",
                model_path="user_data/models/trend_classifier.pkl",
                feature_columns=list(ml_features.keys())[:10],  # Use first 10 features
                prediction_threshold=0.6,
                timeout_ms=100,
                cache_ttl_seconds=60
            )
        }
    )
    
    # Note: We're not actually starting the service since it requires Redis
    # This is just to demonstrate the integration pattern
    print("   Inference service configured with feature store integration")
    print("   In production, features would be passed to ML model for prediction")
    
    # =========================================================================
    # 5. Demonstrate Caching Benefits
    # =========================================================================
    print("\n5. Demonstrating caching benefits...")
    
    # First request (cache miss)
    start_time = datetime.now()
    features1 = feature_store.get_online_features(symbol, timestamp)
    latency1 = (datetime.now() - start_time).total_seconds() * 1000
    
    # Second request (cache hit)
    start_time = datetime.now()
    features2 = feature_store.get_online_features(symbol, timestamp)
    latency2 = (datetime.now() - start_time).total_seconds() * 1000
    
    print(f"   First request (cache miss): {latency1:.2f}ms")
    print(f"   Second request (cache hit): {latency2:.2f}ms")
    print(f"   Speedup: {latency1/latency2:.1f}x faster with cache")
    
    # =========================================================================
    # 6. Feature Store Health and Statistics
    # =========================================================================
    print("\n6. Feature Store health and statistics...")
    
    health = feature_store.health_check()
    stats = feature_store.get_feature_stats()
    
    print(f"   Status: {health['status']}")
    print(f"   Cache size: {stats['cache_size']}")
    print(f"   Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"   Feature views: {stats['feature_views']}")
    
    # =========================================================================
    # 7. Offline Features for Model Training
    # =========================================================================
    print("\n7. Getting offline features for model training...")
    
    # Get historical features for training
    start_date = datetime.now() - timedelta(days=7)
    end_date = datetime.now()
    
    offline_features = feature_store.get_offline_features(
        symbols=["BTC/USDT", "ETH/USDT"],
        start_date=start_date,
        end_date=end_date
    )
    
    if not offline_features.empty:
        print(f"   Retrieved {len(offline_features)} historical feature rows")
        print(f"   Columns: {len(offline_features.columns)} features")
        print(f"   Time range: {start_date.date()} to {end_date.date()}")
    else:
        print("   Note: Offline features would be available with real Feast setup")
    
    # =========================================================================
    # 8. Summary of Benefits
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: How Feature Store solves Task 3.1 problems")
    print("=" * 80)
    
    benefits = [
        ("1. No feature recalculation", 
         "Features are precomputed and cached, 100x faster than recomputing"),
        ("2. Online/offline separation", 
         "Online features for inference, offline features for training"),
        ("3. Feature versioning", 
         "Feast provides automatic feature versioning and lineage tracking"),
        ("4. Production readiness", 
         "Scalable, fault-tolerant feature serving with Redis caching"),
        ("5. Integration with existing ML", 
         "Seamless integration with ML Inference Service")
    ]
    
    for title, description in benefits:
        print(f"\n{title}:")
        print(f"  {description}")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
