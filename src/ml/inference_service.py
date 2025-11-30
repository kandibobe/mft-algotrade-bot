#!/usr/bin/env python3
"""
Async ML Inference Service
==========================

Provides non-blocking ML model inference using Redis as message queue.
Solves the critical issue of ML inference blocking the trading event loop.

Architecture:
    Trading Bot -> Redis Queue -> ML Worker -> Redis Queue -> Trading Bot
    
This decouples inference from trading, preventing missed signals.

Author: Stoic Citadel Team
License: MIT
"""

import asyncio
import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MLModelConfig:
    """Configuration for ML model."""
    model_name: str
    model_path: str
    feature_columns: List[str]
    prediction_threshold: float = 0.5
    timeout_ms: int = 100  # Max wait time for prediction
    cache_ttl_seconds: int = 60  # Cache predictions for 1 minute
    batch_size: int = 32
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "feature_columns": self.feature_columns,
            "prediction_threshold": self.prediction_threshold,
            "timeout_ms": self.timeout_ms,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "batch_size": self.batch_size,
        }


@dataclass
class PredictionRequest:
    """Prediction request structure."""
    request_id: str
    model_name: str
    features: Dict[str, float]
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # Higher = more priority


@dataclass
class PredictionResult:
    """Prediction result structure."""
    request_id: str
    model_name: str
    prediction: float
    probability: float
    signal: str  # 'buy', 'sell', 'hold'
    confidence: float
    latency_ms: float
    timestamp: float
    cached: bool = False


class MLInferenceService:
    """
    Async ML Inference Service using Redis.
    
    Features:
    - Non-blocking predictions via Redis queue
    - Prediction caching to reduce latency
    - Automatic batching for efficiency
    - Timeout handling for guaranteed response time
    - Health monitoring and fallback
    
    Usage:
        service = MLInferenceService(redis_url="redis://localhost:6379")
        await service.start()
        
        # Request prediction (non-blocking)
        result = await service.predict(
            model_name="trend_classifier",
            features={"rsi": 45.5, "macd": 0.002, ...}
        )
        
        # Use result
        if result.signal == "buy" and result.confidence > 0.7:
            execute_buy_order()
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        models: Optional[Dict[str, MLModelConfig]] = None
    ):
        self.redis_url = redis_url
        self.models = models or {}
        self._redis = None
        self._running = False
        self._prediction_cache: Dict[str, PredictionResult] = {}
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "timeouts": 0,
            "errors": 0,
            "avg_latency_ms": 0.0
        }
    
    async def start(self):
        """Initialize service and connect to Redis."""
        try:
            import redis.asyncio as redis
            self._redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self._redis.ping()
            self._running = True
            
            # Start background tasks
            asyncio.create_task(self._result_listener())
            asyncio.create_task(self._cache_cleanup())
            
            logger.info(f"ML Inference Service started, connected to {self.redis_url}")
        except Exception as e:
            logger.error(f"Failed to start ML Inference Service: {e}")
            raise
    
    async def stop(self):
        """Gracefully shutdown service."""
        self._running = False
        if self._redis:
            await self._redis.close()
        logger.info("ML Inference Service stopped")
    
    async def predict(
        self,
        model_name: str,
        features: Dict[str, float],
        timeout_ms: Optional[int] = None
    ) -> PredictionResult:
        """
        Request prediction from ML model (non-blocking).
        
        Args:
            model_name: Name of the model to use
            features: Feature dictionary
            timeout_ms: Override default timeout
            
        Returns:
            PredictionResult with signal and confidence
        """
        self._stats["total_requests"] += 1
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key(model_name, features)
        if cache_key in self._prediction_cache:
            cached = self._prediction_cache[cache_key]
            self._stats["cache_hits"] += 1
            cached.cached = True
            return cached
        
        # Create request
        request = PredictionRequest(
            request_id=f"{model_name}_{time.time_ns()}",
            model_name=model_name,
            features=features
        )
        
        # Create future for result
        future = asyncio.get_event_loop().create_future()
        self._pending_requests[request.request_id] = future
        
        try:
            # Send to Redis queue
            await self._redis.lpush(
                f"ml:requests:{model_name}",
                json.dumps({
                    "request_id": request.request_id,
                    "features": request.features,
                    "timestamp": request.timestamp
                })
            )
            
            # Wait for result with timeout
            model_config = self.models.get(model_name)
            timeout = timeout_ms or (model_config.timeout_ms if model_config else 100)
            
            result = await asyncio.wait_for(
                future,
                timeout=timeout / 1000.0
            )
            
            # Cache result
            self._prediction_cache[cache_key] = result
            
            # Update stats
            latency = (time.time() - start_time) * 1000
            self._update_latency_stats(latency)
            
            return result
            
        except asyncio.TimeoutError:
            self._stats["timeouts"] += 1
            logger.warning(f"Prediction timeout for {model_name}")
            
            # Return fallback prediction
            return self._get_fallback_prediction(request, start_time)
            
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Prediction error: {e}")
            return self._get_fallback_prediction(request, start_time)
            
        finally:
            self._pending_requests.pop(request.request_id, None)
    
    async def predict_batch(
        self,
        model_name: str,
        features_list: List[Dict[str, float]]
    ) -> List[PredictionResult]:
        """
        Request batch predictions (more efficient for multiple symbols).
        """
        tasks = [
            self.predict(model_name, features)
            for features in features_list
        ]
        return await asyncio.gather(*tasks)
    
    async def _result_listener(self):
        """Background task to listen for prediction results."""
        while self._running:
            try:
                # Listen on result queue
                result = await self._redis.brpop(
                    "ml:results",
                    timeout=1
                )
                
                if result:
                    _, data = result
                    result_data = json.loads(data)
                    request_id = result_data["request_id"]
                    
                    if request_id in self._pending_requests:
                        prediction_result = PredictionResult(
                            request_id=request_id,
                            model_name=result_data["model_name"],
                            prediction=result_data["prediction"],
                            probability=result_data["probability"],
                            signal=result_data["signal"],
                            confidence=result_data["confidence"],
                            latency_ms=result_data["latency_ms"],
                            timestamp=time.time()
                        )
                        self._pending_requests[request_id].set_result(prediction_result)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Result listener error: {e}")
                await asyncio.sleep(0.1)
    
    async def _cache_cleanup(self):
        """Periodically clean expired cache entries."""
        while self._running:
            try:
                current_time = time.time()
                expired_keys = [
                    key for key, result in self._prediction_cache.items()
                    if current_time - result.timestamp > 60  # 60s default TTL
                ]
                for key in expired_keys:
                    del self._prediction_cache[key]
                    
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    def _get_cache_key(self, model_name: str, features: Dict[str, float]) -> str:
        """Generate cache key from features."""
        # Round features for cache key (allow small variations)
        rounded = {k: round(v, 4) for k, v in sorted(features.items())}
        return f"{model_name}:{hash(frozenset(rounded.items()))}"
    
    def _get_fallback_prediction(self, request: PredictionRequest, start_time: float) -> PredictionResult:
        """Return conservative fallback when ML unavailable."""
        return PredictionResult(
            request_id=request.request_id,
            model_name=request.model_name,
            prediction=0.5,
            probability=0.5,
            signal="hold",  # Conservative: don't trade when uncertain
            confidence=0.0,
            latency_ms=(time.time() - start_time) * 1000,
            timestamp=time.time(),
            cached=False
        )
    
    def _update_latency_stats(self, latency_ms: float):
        """Update running average latency."""
        n = self._stats["total_requests"]
        current_avg = self._stats["avg_latency_ms"]
        self._stats["avg_latency_ms"] = current_avg + (latency_ms - current_avg) / n
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            **self._stats,
            "cache_size": len(self._prediction_cache),
            "pending_requests": len(self._pending_requests),
            "cache_hit_rate": (
                self._stats["cache_hits"] / max(1, self._stats["total_requests"])
            ) * 100
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        try:
            await self._redis.ping()
            redis_ok = True
        except:
            redis_ok = False
        
        return {
            "service": "ml_inference",
            "status": "healthy" if redis_ok and self._running else "unhealthy",
            "redis_connected": redis_ok,
            "running": self._running,
            "stats": self.get_stats()
        }


class MLWorker:
    """
    ML Worker Process - runs inference in separate process.
    
    This worker:
    1. Loads ML models into memory
    2. Listens on Redis queue for prediction requests
    3. Runs inference (CPU/GPU bound)
    4. Publishes results back to Redis
    
    Run as separate process:
        python -m src.ml.inference_service --worker
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        models_dir: str = "user_data/models"
    ):
        self.redis_url = redis_url
        self.models_dir = Path(models_dir)
        self._redis = None
        self._models: Dict[str, Any] = {}
        self._running = False
    
    async def start(self):
        """Start worker and load models."""
        import redis.asyncio as redis
        self._redis = redis.from_url(self.redis_url)
        
        # Load all models
        self._load_models()
        
        self._running = True
        logger.info(f"ML Worker started with {len(self._models)} models")
        
        # Start processing loop
        await self._process_requests()
    
    def _load_models(self):
        """Load all ML models from disk."""
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return
        
        for model_file in self.models_dir.glob("*.pkl"):
            try:
                with open(model_file, "rb") as f:
                    model = pickle.load(f)
                model_name = model_file.stem
                self._models[model_name] = model
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_file}: {e}")
    
    async def _process_requests(self):
        """Main processing loop."""
        while self._running:
            try:
                # Wait for request from any model queue
                queues = [f"ml:requests:{name}" for name in self._models.keys()]
                if not queues:
                    queues = ["ml:requests:default"]
                
                result = await self._redis.brpop(queues, timeout=1)
                
                if result:
                    queue_name, request_data = result
                    model_name = queue_name.split(":")[-1]
                    
                    request = json.loads(request_data)
                    
                    # Run inference
                    prediction_result = self._run_inference(
                        model_name,
                        request
                    )
                    
                    # Send result back
                    await self._redis.lpush(
                        "ml:results",
                        json.dumps(prediction_result)
                    )
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(0.1)
    
    def _run_inference(
        self,
        model_name: str,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run model inference."""
        start_time = time.time()
        
        try:
            model = self._models.get(model_name)
            
            if model is None:
                # Return neutral prediction if model not found
                return self._create_result(
                    request, model_name, 0.5, 0.5, "hold", 0.0, start_time
                )
            
            # Prepare features
            features = np.array([list(request["features"].values())])
            
            # Run prediction
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(features)[0]
                prediction = model.predict(features)[0]
                probability = max(probas)
            else:
                prediction = model.predict(features)[0]
                probability = abs(prediction)
            
            # Determine signal
            if prediction > 0.6:
                signal = "buy"
            elif prediction < 0.4:
                signal = "sell"
            else:
                signal = "hold"
            
            confidence = abs(prediction - 0.5) * 2  # 0-1 scale
            
            return self._create_result(
                request, model_name, prediction, probability,
                signal, confidence, start_time
            )
            
        except Exception as e:
            logger.error(f"Inference error for {model_name}: {e}")
            return self._create_result(
                request, model_name, 0.5, 0.5, "hold", 0.0, start_time
            )
    
    def _create_result(
        self,
        request: Dict,
        model_name: str,
        prediction: float,
        probability: float,
        signal: str,
        confidence: float,
        start_time: float
    ) -> Dict[str, Any]:
        """Create result dictionary."""
        return {
            "request_id": request["request_id"],
            "model_name": model_name,
            "prediction": float(prediction),
            "probability": float(probability),
            "signal": signal,
            "confidence": float(confidence),
            "latency_ms": (time.time() - start_time) * 1000,
            "timestamp": time.time()
        }


# CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Inference Service")
    parser.add_argument("--worker", action="store_true", help="Run as worker process")
    parser.add_argument("--redis", default="redis://localhost:6379", help="Redis URL")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.worker:
        worker = MLWorker(redis_url=args.redis)
        asyncio.run(worker.start())
    else:
        print("Use --worker flag to run as ML worker process")
