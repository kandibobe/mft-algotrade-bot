"""
Stoic Citadel - Model Registry
==============================

ML model versioning and management:
- Model registration and versioning
- Performance tracking
- Model deployment lifecycle
- A/B testing support
"""

import os
import json
import hashlib
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import pickle

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model lifecycle stages."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    backtest_period_days: int = 0
    validation_date: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ModelVersion:
    """Model version metadata."""
    version: str
    model_name: str
    created_at: str
    model_type: str  # e.g., "lightgbm", "xgboost", "pytorch"
    
    # Training info
    training_data_hash: str = ""
    training_params: Dict = field(default_factory=dict)
    feature_columns: List[str] = field(default_factory=list)
    
    # Performance
    metrics: ModelMetrics = field(default_factory=ModelMetrics)
    
    # Lifecycle
    stage: str = ModelStage.DEVELOPMENT.value
    promoted_at: Optional[str] = None
    promoted_by: Optional[str] = None
    
    # Files
    model_path: str = ""
    artifacts_path: str = ""
    
    # Description
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d["metrics"] = self.metrics.to_dict() if isinstance(self.metrics, ModelMetrics) else self.metrics
        return d
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ModelVersion":
        metrics_data = data.pop("metrics", {})
        if isinstance(metrics_data, dict):
            data["metrics"] = ModelMetrics(**metrics_data)
        return cls(**data)


class ModelRegistry:
    """
    Local model registry for versioning and management.
    
    Features:
    - Model versioning (semantic versioning)
    - Stage management (dev -> staging -> production)
    - Performance tracking
    - Rollback support
    """
    
    def __init__(
        self,
        registry_path: str = "./model_registry",
        max_versions_per_model: int = 10
    ):
        self.registry_path = Path(registry_path)
        self.max_versions = max_versions_per_model
        self._models: Dict[str, Dict[str, ModelVersion]] = {}
        self._init_registry()
    
    def _init_registry(self) -> None:
        """Initialize registry directory structure."""
        self.registry_path.mkdir(parents=True, exist_ok=True)
        (self.registry_path / "models").mkdir(exist_ok=True)
        (self.registry_path / "metadata").mkdir(exist_ok=True)
        
        # Load existing registry
        self._load_registry()
    
    def register_model(
        self,
        model_name: str,
        model_object: Any,
        model_type: str,
        training_params: Dict,
        feature_columns: List[str],
        metrics: Optional[ModelMetrics] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        training_data: Optional[Any] = None
    ) -> ModelVersion:
        """
        Register a new model version.
        
        Args:
            model_name: Name of the model
            model_object: Trained model object
            model_type: Type of model (lightgbm, xgboost, etc.)
            training_params: Training hyperparameters
            feature_columns: List of feature column names
            metrics: Performance metrics from validation
            description: Model description
            tags: Tags for categorization
            training_data: Training data for hash computation
            
        Returns:
            ModelVersion object
        """
        # Generate version number
        version = self._get_next_version(model_name)
        
        # Create version directory
        version_dir = self.registry_path / "models" / model_name / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = version_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_object, f)
        
        # Compute training data hash
        data_hash = ""
        if training_data is not None:
            data_hash = self._compute_data_hash(training_data)
        
        # Create version metadata
        model_version = ModelVersion(
            version=version,
            model_name=model_name,
            created_at=datetime.utcnow().isoformat(),
            model_type=model_type,
            training_data_hash=data_hash,
            training_params=training_params,
            feature_columns=feature_columns,
            metrics=metrics or ModelMetrics(),
            stage=ModelStage.DEVELOPMENT.value,
            model_path=str(model_path),
            artifacts_path=str(version_dir),
            description=description,
            tags=tags or []
        )
        
        # Save metadata
        self._save_version_metadata(model_version)
        
        # Update in-memory registry
        if model_name not in self._models:
            self._models[model_name] = {}
        self._models[model_name][version] = model_version
        
        # Cleanup old versions
        self._cleanup_old_versions(model_name)
        
        logger.info(f"Registered model: {model_name} v{version}")
        
        return model_version
    
    def get_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[ModelStage] = None
    ) -> Any:
        """
        Load a model from registry.
        
        Args:
            model_name: Name of the model
            version: Specific version (optional)
            stage: Get model at specific stage (optional)
            
        Returns:
            Loaded model object
        """
        model_version = self.get_model_version(model_name, version, stage)
        
        if model_version is None:
            raise ValueError(f"Model not found: {model_name}")
        
        with open(model_version.model_path, 'rb') as f:
            return pickle.load(f)
    
    def get_model_version(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[ModelStage] = None
    ) -> Optional[ModelVersion]:
        """
        Get model version metadata.
        """
        if model_name not in self._models:
            return None
        
        versions = self._models[model_name]
        
        if version:
            return versions.get(version)
        
        if stage:
            # Find latest version at stage
            stage_versions = [
                v for v in versions.values()
                if v.stage == stage.value
            ]
            if stage_versions:
                return max(stage_versions, key=lambda x: x.created_at)
            return None
        
        # Return latest version
        return max(versions.values(), key=lambda x: x.created_at)
    
    def promote_model(
        self,
        model_name: str,
        version: str,
        to_stage: ModelStage,
        promoted_by: str = "system"
    ) -> None:
        """
        Promote model to new stage.
        
        When promoting to production, demotes current production model.
        """
        model_version = self.get_model_version(model_name, version)
        if model_version is None:
            raise ValueError(f"Model version not found: {model_name} v{version}")
        
        # Demote current production if promoting to production
        if to_stage == ModelStage.PRODUCTION:
            current_prod = self.get_model_version(model_name, stage=ModelStage.PRODUCTION)
            if current_prod:
                current_prod.stage = ModelStage.ARCHIVED.value
                self._save_version_metadata(current_prod)
                logger.info(
                    f"Demoted {model_name} v{current_prod.version} to archived"
                )
        
        # Update stage
        model_version.stage = to_stage.value
        model_version.promoted_at = datetime.utcnow().isoformat()
        model_version.promoted_by = promoted_by
        
        self._save_version_metadata(model_version)
        
        logger.info(
            f"Promoted {model_name} v{version} to {to_stage.value}"
        )
    
    def update_metrics(
        self,
        model_name: str,
        version: str,
        metrics: ModelMetrics
    ) -> None:
        """Update model metrics after validation."""
        model_version = self.get_model_version(model_name, version)
        if model_version is None:
            raise ValueError(f"Model version not found: {model_name} v{version}")
        
        model_version.metrics = metrics
        self._save_version_metadata(model_version)
        
        logger.info(f"Updated metrics for {model_name} v{version}")
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self._models.keys())
    
    def list_versions(
        self,
        model_name: str,
        stage: Optional[ModelStage] = None
    ) -> List[ModelVersion]:
        """List versions of a model."""
        if model_name not in self._models:
            return []
        
        versions = list(self._models[model_name].values())
        
        if stage:
            versions = [v for v in versions if v.stage == stage.value]
        
        return sorted(versions, key=lambda x: x.created_at, reverse=True)
    
    def compare_models(
        self,
        model_name: str,
        version1: str,
        version2: str
    ) -> Dict:
        """Compare two model versions."""
        v1 = self.get_model_version(model_name, version1)
        v2 = self.get_model_version(model_name, version2)
        
        if not v1 or not v2:
            raise ValueError("One or both versions not found")
        
        m1 = v1.metrics if isinstance(v1.metrics, ModelMetrics) else ModelMetrics(**v1.metrics)
        m2 = v2.metrics if isinstance(v2.metrics, ModelMetrics) else ModelMetrics(**v2.metrics)
        
        return {
            "versions": [version1, version2],
            "comparison": {
                "sharpe_ratio": {
                    version1: m1.sharpe_ratio,
                    version2: m2.sharpe_ratio,
                    "diff": m2.sharpe_ratio - m1.sharpe_ratio
                },
                "sortino_ratio": {
                    version1: m1.sortino_ratio,
                    version2: m2.sortino_ratio,
                    "diff": m2.sortino_ratio - m1.sortino_ratio
                },
                "max_drawdown": {
                    version1: m1.max_drawdown,
                    version2: m2.max_drawdown,
                    "diff": m2.max_drawdown - m1.max_drawdown
                },
                "win_rate": {
                    version1: m1.win_rate,
                    version2: m2.win_rate,
                    "diff": m2.win_rate - m1.win_rate
                },
                "profit_factor": {
                    version1: m1.profit_factor,
                    version2: m2.profit_factor,
                    "diff": m2.profit_factor - m1.profit_factor
                }
            },
            "recommendation": self._recommend_version(m1, m2, version1, version2)
        }
    
    def delete_version(
        self,
        model_name: str,
        version: str
    ) -> None:
        """Delete a model version."""
        model_version = self.get_model_version(model_name, version)
        if model_version is None:
            return
        
        # Don't delete production models
        if model_version.stage == ModelStage.PRODUCTION.value:
            raise ValueError("Cannot delete production model")
        
        # Remove files
        version_dir = Path(model_version.artifacts_path)
        if version_dir.exists():
            shutil.rmtree(version_dir)
        
        # Remove metadata
        metadata_path = self.registry_path / "metadata" / model_name / f"{version}.json"
        if metadata_path.exists():
            metadata_path.unlink()
        
        # Remove from memory
        del self._models[model_name][version]
        
        logger.info(f"Deleted {model_name} v{version}")
    
    def _get_next_version(self, model_name: str) -> str:
        """Generate next version number."""
        if model_name not in self._models or not self._models[model_name]:
            return "1.0.0"
        
        versions = list(self._models[model_name].keys())
        latest = max(versions, key=lambda v: [int(x) for x in v.split('.')])
        
        parts = [int(x) for x in latest.split('.')]
        parts[-1] += 1
        
        return '.'.join(str(x) for x in parts)
    
    def _compute_data_hash(self, data: Any) -> str:
        """Compute hash of training data."""
        try:
            data_bytes = pickle.dumps(data)
            return hashlib.sha256(data_bytes).hexdigest()[:16]
        except Exception:
            return ""
    
    def _save_version_metadata(self, version: ModelVersion) -> None:
        """Save version metadata to disk."""
        metadata_dir = self.registry_path / "metadata" / version.model_name
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_path = metadata_dir / f"{version.version}.json"
        with open(metadata_path, 'w') as f:
            json.dump(version.to_dict(), f, indent=2)
    
    def _load_registry(self) -> None:
        """Load existing registry from disk."""
        metadata_dir = self.registry_path / "metadata"
        if not metadata_dir.exists():
            return
        
        for model_dir in metadata_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            self._models[model_name] = {}
            
            for version_file in model_dir.glob("*.json"):
                try:
                    with open(version_file) as f:
                        data = json.load(f)
                    version = ModelVersion.from_dict(data)
                    self._models[model_name][version.version] = version
                except Exception as e:
                    logger.error(f"Error loading {version_file}: {e}")
    
    def _cleanup_old_versions(self, model_name: str) -> None:
        """Remove old versions exceeding max limit."""
        if model_name not in self._models:
            return
        
        versions = self._models[model_name]
        
        # Keep production and staging versions
        protected_stages = {ModelStage.PRODUCTION.value, ModelStage.STAGING.value}
        
        # Sort by creation date
        sorted_versions = sorted(
            versions.values(),
            key=lambda x: x.created_at,
            reverse=True
        )
        
        # Find versions to delete
        keep_count = 0
        for v in sorted_versions:
            if v.stage in protected_stages:
                continue
            
            keep_count += 1
            if keep_count > self.max_versions:
                self.delete_version(model_name, v.version)
    
    def _recommend_version(
        self,
        m1: ModelMetrics,
        m2: ModelMetrics,
        v1: str,
        v2: str
    ) -> str:
        """Recommend which version to use."""
        score1 = (
            m1.sharpe_ratio * 0.3 +
            m1.sortino_ratio * 0.2 +
            m1.win_rate * 0.2 +
            m1.profit_factor * 0.2 -
            abs(m1.max_drawdown) * 0.1
        )
        
        score2 = (
            m2.sharpe_ratio * 0.3 +
            m2.sortino_ratio * 0.2 +
            m2.win_rate * 0.2 +
            m2.profit_factor * 0.2 -
            abs(m2.max_drawdown) * 0.1
        )
        
        if score2 > score1 * 1.05:  # 5% improvement threshold
            return f"Recommend {v2} (score: {score2:.3f} vs {score1:.3f})"
        elif score1 > score2 * 1.05:
            return f"Keep {v1} (score: {score1:.3f} vs {score2:.3f})"
        else:
            return f"No significant difference (scores: {score1:.3f} vs {score2:.3f})"
