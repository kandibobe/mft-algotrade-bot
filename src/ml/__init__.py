"""ML Inference Service Module."""

from .inference_service import (
    MLInferenceService, 
    MLModelConfig,
    OptimizedInferenceService,
    MLWorker,
    OptimizedMLWorker,
    PredictionRequest,
    PredictionResult
)
from .redis_client import RedisMLClient
from .feature_store import (
    TradingFeatureStore,
    MockFeatureStore,
    create_feature_store
)
from .meta_learning import (
    MetaLearningEnsemble,
    MetaLearningConfig
)
from .online_learner import (
    OnlineLearner,
    OnlineLearningConfig,
    load_model,
    save_model
)

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
    "create_feature_store",
    "MetaLearningEnsemble",
    "MetaLearningConfig",
    "OnlineLearner",
    "OnlineLearningConfig",
    "load_model",
    "save_model"
]
