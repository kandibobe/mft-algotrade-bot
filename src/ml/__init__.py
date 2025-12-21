"""ML Inference Service Module."""

from .feature_store import (
    MockFeatureStore,
    RedisFeatureStore,
    TradingFeatureStore,
    create_feature_store,
)
from .inference_service import (
    MLInferenceService,
    MLModelConfig,
    MLWorker,
    OptimizedInferenceService,
    OptimizedMLWorker,
    PredictionRequest,
    PredictionResult,
)
from .meta_learning import MetaLearningConfig, MetaLearningEnsemble
from .online_learner import OnlineLearner, OnlineLearningConfig, load_model, save_model
from .redis_client import RedisMLClient

__all__ = [
    "MLInferenceService",
    "MLModelConfig",
    "OptimizedInferenceService",
    "MLWorker",
    "OptimizedMLWorker",
    "PredictionRequest",
    "PredictionResult",
    "RedisMLClient",
    "TradingFeatureStore",
    "MockFeatureStore",
    "RedisFeatureStore",
    "create_feature_store",
    "MetaLearningEnsemble",
    "MetaLearningConfig",
    "OnlineLearner",
    "OnlineLearningConfig",
    "load_model",
    "save_model",
]
