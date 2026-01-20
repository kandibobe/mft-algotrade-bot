"""
Model Registry
==============

Manage ML model versions and deployment.
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from src.config import config

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model status in registry."""

    STAGED = "staged"  # In testing
    PRODUCTION = "production"  # Active in production
    ARCHIVED = "archived"  # No longer in use
    FAILED = "failed"  # Failed validation


@dataclass
class ModelMetadata:
    """Metadata for registered model."""

    name: str
    version: str
    model_type: str
    status: ModelStatus = ModelStatus.STAGED

    # Paths
    model_path: str = ""
    metadata_path: str = ""

    # Performance metrics
    metrics: dict[str, float] = field(default_factory=dict)
    backtest_results: dict[str, Any] = field(default_factory=dict)

    # Training info
    trained_at: datetime = field(default_factory=datetime.now)
    trained_by: str = "unknown"
    training_config: dict[str, Any] = field(default_factory=dict)

    # Feature info
    feature_count: int = 0
    feature_names: list[str] = field(default_factory=list)

    # Validation
    validation_passed: bool = False
    validation_notes: str = ""

    # Deployment
    deployed_at: datetime | None = None
    deployment_notes: str = ""

    # Tags
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type,
            "status": self.status.value,
            "model_path": self.model_path,
            "metrics": self.metrics,
            "backtest_results": self.backtest_results,
            "trained_at": self.trained_at.isoformat(),
            "trained_by": self.trained_by,
            "training_config": self.training_config,
            "feature_count": self.feature_count,
            "feature_names": self.feature_names,
            "validation_passed": self.validation_passed,
            "validation_notes": self.validation_notes,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "deployment_notes": self.deployment_notes,
            "tags": self.tags,
        }


class ModelRegistry:
    """
    Model registry for version management and deployment.

    Features:
    - Register new models
    - Track model versions
    - Validate models before production
    - Promote models to production
    - Rollback to previous versions
    - Archive old models

    Usage:
        registry = ModelRegistry()

        # Register model
        registry.register_model(
            model_name="trend_classifier",
            model_path="models/rf_20250101.pkl",
            metrics={"f1": 0.85},
            backtest_results={"sharpe": 1.8}
        )

        # Validate and promote
        if registry.validate_model("trend_classifier", "v1.0"):
            registry.promote_to_production("trend_classifier", "v1.0")
    """

    def __init__(self, registry_dir: str | None = None):
        """
        Initialize model registry.

        Args:
            registry_dir: Directory to store model registry files
        """
        self.registry_dir = Path(registry_dir or config().paths.models_dir / "registry")
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.registry_file = self.registry_dir / "registry.json"
        self.models: dict[str, list[ModelMetadata]] = {}

        # Load existing registry
        self._load_registry()

    def register_model(
        self,
        model_name: str,
        model_path: str,
        version: str | None = None,
        metrics: dict[str, float] | None = None,
        backtest_results: dict[str, Any] | None = None,
        training_config: dict[str, Any] | None = None,
        feature_names: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> ModelMetadata:
        """
        Register a new model.

        Args:
            model_name: Model name
            model_path: Path to model file
            version: Version string (auto-generated if not provided)
            metrics: Training/validation metrics
            backtest_results: Backtest results
            training_config: Training configuration
            feature_names: List of feature names
            tags: Model tags

        Returns:
            ModelMetadata object
        """
        # Auto-generate version if not provided
        if version is None:
            version = self._generate_version(model_name)

        # Check if version already exists
        if model_name in self.models:
            for model in self.models[model_name]:
                if model.version == version:
                    logger.warning(
                        f"Model {model_name} version {version} already exists. "
                        "Use a different version or update existing."
                    )
                    return model

        # Create metadata
        model_path_obj = Path(model_path)
        model_type = model_path_obj.stem.split("_")[0] if "_" in model_path_obj.stem else "unknown"

        metadata = ModelMetadata(
            name=model_name,
            version=version,
            model_type=model_type,
            model_path=str(model_path),
            metrics=metrics or {},
            backtest_results=backtest_results or {},
            training_config=training_config or {},
            feature_names=feature_names or [],
            feature_count=len(feature_names) if feature_names else 0,
            tags=tags or [],
        )

        # Add to registry
        if model_name not in self.models:
            self.models[model_name] = []
        self.models[model_name].append(metadata)

        # Save registry
        self._save_registry()

        logger.info(
            f"Registered model: {model_name} v{version} "
            f"(F1: {metrics.get('f1', 'N/A') if metrics else 'N/A'})"
        )

        return metadata

    def validate_model(
        self,
        model_name: str,
        version: str,
        min_metrics: dict[str, float] | None = None,
        min_backtest_sharpe: float = 1.0,
        min_backtest_trades: int = 10,
    ) -> bool:
        """
        Validate model before promotion to production.

        Args:
            model_name: Model name
            version: Version to validate
            min_metrics: Minimum required metrics
            min_backtest_sharpe: Minimum Sharpe ratio
            min_backtest_trades: Minimum number of trades

        Returns:
            True if validation passed
        """
        metadata = self._get_model(model_name, version)
        if not metadata:
            logger.error(f"Model {model_name} v{version} not found")
            return False

        validation_errors = []

        # Check metrics
        if min_metrics:
            for metric, min_value in min_metrics.items():
                actual_value = metadata.metrics.get(metric, 0)
                if actual_value < min_value:
                    validation_errors.append(f"{metric}: {actual_value:.4f} < {min_value:.4f}")

        # Check backtest results
        if metadata.backtest_results:
            sharpe = metadata.backtest_results.get("sharpe_ratio", 0)
            if sharpe < min_backtest_sharpe:
                validation_errors.append(f"Sharpe ratio: {sharpe:.2f} < {min_backtest_sharpe:.2f}")

            trades = metadata.backtest_results.get("total_trades", 0)
            if trades < min_backtest_trades:
                validation_errors.append(f"Total trades: {trades} < {min_backtest_trades}")

        # Update metadata
        if validation_errors:
            metadata.validation_passed = False
            metadata.validation_notes = "; ".join(validation_errors)
            metadata.status = ModelStatus.FAILED
            self._save_registry()

            logger.error(
                f"Validation FAILED for {model_name} v{version}:\n"
                + "\n".join(f"  - {err}" for err in validation_errors)
            )
            return False
        else:
            metadata.validation_passed = True
            metadata.validation_notes = "All validation checks passed"
            self._save_registry()

            logger.info(f"Validation PASSED for {model_name} v{version}")
            return True

    def promote_to_production(self, model_name: str, version: str, notes: str = "") -> bool:
        """
        Promote model to production.

        Args:
            model_name: Model name
            version: Version to promote
            notes: Deployment notes

        Returns:
            True if promoted successfully
        """
        metadata = self._get_model(model_name, version)
        if not metadata:
            logger.error(f"Model {model_name} v{version} not found")
            return False

        # Check if validated
        if not metadata.validation_passed:
            logger.error(
                f"Cannot promote {model_name} v{version} - validation not passed. "
                "Run validate_model() first."
            )
            return False

        # Demote current production model
        if model_name in self.models:
            for model in self.models[model_name]:
                if model.status == ModelStatus.PRODUCTION:
                    model.status = ModelStatus.ARCHIVED
                    logger.info(f"Archived previous production model: v{model.version}")

        # Promote new model
        metadata.status = ModelStatus.PRODUCTION
        metadata.deployed_at = datetime.now()
        metadata.deployment_notes = notes

        # Create symlink to production model
        self._create_production_symlink(metadata)

        self._save_registry()

        logger.info(f"✅ Promoted {model_name} v{version} to PRODUCTION")

        return True

    def rollback_to_version(self, model_name: str, version: str) -> bool:
        """
        Rollback to previous model version.

        Args:
            model_name: Model name
            version: Version to rollback to

        Returns:
            True if rollback successful
        """
        logger.warning(f"Rolling back {model_name} to v{version}")

        return self.promote_to_production(model_name, version, notes="Rolled back from production")

    def get_production_model(self, model_name: str) -> ModelMetadata | None:
        """
        Get current production model.

        Args:
            model_name: Model name

        Returns:
            ModelMetadata or None
        """
        if model_name not in self.models:
            return None

        for model in self.models[model_name]:
            if model.status == ModelStatus.PRODUCTION:
                return model

        return None

    def get_all_versions(self, model_name: str) -> list[ModelMetadata]:
        """
        Get all versions of a model.

        Args:
            model_name: Model name

        Returns:
            List of ModelMetadata sorted by version (newest first)
        """
        if model_name not in self.models:
            return []

        return sorted(self.models[model_name], key=lambda m: m.trained_at, reverse=True)

    def get_feature_importance(self, model_name: str, version: str) -> dict[str, float]:
        """
        Get feature importance for a specific model version.

        Args:
            model_name: Model name
            version: Model version

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        metadata = self._get_model(model_name, version)
        if not metadata or not metadata.model_path:
            return {}

        import pandas as pd

        # Look for importance CSV next to model file
        model_path = Path(metadata.model_path)
        importance_path = model_path.with_suffix(".csv")

        if not importance_path.exists():
            logger.warning(f"Feature importance file not found: {importance_path}")
            return {}

        try:
            df = pd.read_csv(importance_path)
            # Assuming columns 'feature' and 'importance'
            if "feature" in df.columns and "importance" in df.columns:
                return dict(zip(df["feature"], df["importance"], strict=False))
            return {}
        except Exception as e:
            logger.error(f"Failed to load feature importance: {e}")
            return {}

    def get_model_history(self, model_name: str) -> dict[str, Any]:
        """
        Get model version history.

        Args:
            model_name: Model name

        Returns:
            Dictionary with version history
        """
        versions = self.get_all_versions(model_name)

        return {
            "model_name": model_name,
            "total_versions": len(versions),
            "production_version": next(
                (v.version for v in versions if v.status == ModelStatus.PRODUCTION), None
            ),
            "versions": [
                {
                    "version": v.version,
                    "status": v.status.value,
                    "trained_at": v.trained_at.isoformat(),
                    "metrics": v.metrics,
                    "validated": v.validation_passed,
                }
                for v in versions
            ],
        }

    def archive_model(self, model_name: str, version: str):
        """
        Archive a model version.

        Args:
            model_name: Model name
            version: Version to archive
        """
        metadata = self._get_model(model_name, version)
        if metadata:
            if metadata.status == ModelStatus.PRODUCTION:
                logger.error("Cannot archive production model. Promote another version first.")
                return

            metadata.status = ModelStatus.ARCHIVED
            self._save_registry()

            logger.info(f"Archived {model_name} v{version}")

    def delete_model(self, model_name: str, version: str, delete_files: bool = False):
        """
        Delete a model version.

        Args:
            model_name: Model name
            version: Version to delete
            delete_files: Also delete model files from disk
        """
        if model_name not in self.models:
            return

        metadata = self._get_model(model_name, version)
        if not metadata:
            return

        if metadata.status == ModelStatus.PRODUCTION:
            logger.error("Cannot delete production model!")
            return

        # Delete files if requested
        if delete_files and metadata.model_path:
            try:
                Path(metadata.model_path).unlink()
                logger.info(f"Deleted model file: {metadata.model_path}")
            except Exception as e:
                logger.error(f"Failed to delete model file: {e}")

        # Remove from registry
        self.models[model_name] = [m for m in self.models[model_name] if m.version != version]

        self._save_registry()

        logger.info(f"Deleted {model_name} v{version} from registry")

    def _get_model(self, model_name: str, version: str) -> ModelMetadata | None:
        """Get model metadata by name and version."""
        if model_name not in self.models:
            return None

        for model in self.models[model_name]:
            if model.version == version:
                return model

        return None

    def _generate_version(self, model_name: str) -> str:
        """Generate next version number."""
        if model_name not in self.models or not self.models[model_name]:
            return "v1.0"

        # Find highest version
        versions = [m.version for m in self.models[model_name]]
        major_versions = []
        for v in versions:
            try:
                if v.startswith("v"):
                    major = int(v.split(".")[0][1:])
                    major_versions.append(major)
            except:
                pass

        if major_versions:
            next_major = max(major_versions) + 1
        else:
            next_major = 1

        return f"v{next_major}.0"

    def _create_production_symlink(self, metadata: ModelMetadata):
        """Create symlink to production model."""
        if not metadata.model_path:
            return

        model_path = Path(metadata.model_path)
        if not model_path.exists():
            return

        # Create symlink
        production_link = model_path.parent / f"{metadata.name}_production{model_path.suffix}"

        try:
            if production_link.exists() or production_link.is_symlink():
                production_link.unlink()

            # On Windows, copy file instead of symlink (requires admin for symlinks)
            import platform

            if platform.system() == "Windows":
                shutil.copy2(model_path, production_link)
            else:
                production_link.symlink_to(model_path)

            logger.info(f"Created production link: {production_link}")
        except Exception as e:
            logger.warning(f"Could not create production link: {e}")

    def _load_registry(self):
        """Load registry from disk."""
        if not self.registry_file.exists():
            return

        try:
            with open(self.registry_file) as f:
                data = json.load(f)

            for model_name, versions in data.items():
                self.models[model_name] = []
                for v in versions:
                    metadata = ModelMetadata(
                        name=v["name"],
                        version=v["version"],
                        model_type=v["model_type"],
                        status=ModelStatus(v["status"]),
                        model_path=v["model_path"],
                        metrics=v.get("metrics", {}),
                        backtest_results=v.get("backtest_results", {}),
                        trained_at=datetime.fromisoformat(v["trained_at"]),
                        training_config=v.get("training_config", {}),
                        feature_count=v.get("feature_count", 0),
                        feature_names=v.get("feature_names", []),
                        validation_passed=v.get("validation_passed", False),
                        validation_notes=v.get("validation_notes", ""),
                        deployed_at=(
                            datetime.fromisoformat(v["deployed_at"])
                            if v.get("deployed_at")
                            else None
                        ),
                        deployment_notes=v.get("deployment_notes", ""),
                        tags=v.get("tags", []),
                    )
                    self.models[model_name].append(metadata)

            logger.info(f"Loaded registry with {len(self.models)} models")

        except Exception as e:
            logger.error(f"Failed to load registry: {e}")

    def _save_registry(self):
        """Save registry to disk atomically."""
        data = {}
        for model_name, versions in self.models.items():
            data[model_name] = [v.to_dict() for v in versions]

        # Atomic write pattern to prevent corruption
        temp_file = self.registry_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)

            # Atomic rename (replace)
            temp_file.replace(self.registry_file)
            logger.debug(f"Saved registry to {self.registry_file}")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            if temp_file.exists():
                temp_file.unlink()
