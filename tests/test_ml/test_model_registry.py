"""
Tests for Model Registry
"""

import tempfile
from pathlib import Path

import pytest

from src.ml.training.model_registry import (
    ModelMetadata,
    ModelRegistry,
    ModelStatus,
)


@pytest.fixture
def temp_registry_dir():
    """Create temporary registry directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_model_file(temp_registry_dir):
    """Create sample model file."""
    model_path = Path(temp_registry_dir) / "models" / "test_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Create dummy file
    model_path.write_text("dummy model content")

    return str(model_path)


class TestModelMetadata:
    """Test ModelMetadata dataclass."""

    def test_initialization(self):
        """Test metadata initialization."""
        metadata = ModelMetadata(
            name="test_model",
            version="v1.0",
            model_type="random_forest",
        )

        assert metadata.name == "test_model"
        assert metadata.version == "v1.0"
        assert metadata.model_type == "random_forest"
        assert metadata.status == ModelStatus.STAGED
        assert metadata.validation_passed is False

    def test_to_dict(self):
        """Test metadata serialization."""
        metadata = ModelMetadata(
            name="test_model",
            version="v1.0",
            model_type="xgboost",
            metrics={"f1": 0.85, "accuracy": 0.82},
        )

        result = metadata.to_dict()

        assert isinstance(result, dict)
        assert result["name"] == "test_model"
        assert result["version"] == "v1.0"
        assert result["metrics"]["f1"] == 0.85


class TestModelRegistry:
    """Test ModelRegistry class."""

    def test_initialization(self, temp_registry_dir):
        """Test registry initialization."""
        registry = ModelRegistry(registry_dir=temp_registry_dir)

        assert registry.registry_dir.exists()
        assert registry.models == {}

    def test_register_model(self, temp_registry_dir, sample_model_file):
        """Test model registration."""
        registry = ModelRegistry(registry_dir=temp_registry_dir)

        metadata = registry.register_model(
            model_name="trend_classifier",
            model_path=sample_model_file,
            version="v1.0",
            metrics={"f1": 0.85},
        )

        # Check metadata
        assert metadata.name == "trend_classifier"
        assert metadata.version == "v1.0"
        assert metadata.metrics["f1"] == 0.85
        assert metadata.status == ModelStatus.STAGED

        # Check registry contains model
        assert "trend_classifier" in registry.models
        assert len(registry.models["trend_classifier"]) == 1

    def test_register_duplicate_version(self, temp_registry_dir, sample_model_file):
        """Test registering duplicate version."""
        registry = ModelRegistry(registry_dir=temp_registry_dir)

        # Register first time
        metadata1 = registry.register_model(
            model_name="test_model",
            model_path=sample_model_file,
            version="v1.0",
        )

        # Register same version again
        metadata2 = registry.register_model(
            model_name="test_model",
            model_path=sample_model_file,
            version="v1.0",
        )

        # Should return existing metadata
        assert metadata1.version == metadata2.version

    def test_auto_version_generation(self, temp_registry_dir, sample_model_file):
        """Test automatic version generation."""
        registry = ModelRegistry(registry_dir=temp_registry_dir)

        # Register without version
        metadata1 = registry.register_model(
            model_name="test_model",
            model_path=sample_model_file,
        )

        assert metadata1.version == "v1.0"

        # Register another
        metadata2 = registry.register_model(
            model_name="test_model",
            model_path=sample_model_file,
        )

        assert metadata2.version == "v2.0"

    def test_validate_model_success(self, temp_registry_dir, sample_model_file):
        """Test successful model validation."""
        registry = ModelRegistry(registry_dir=temp_registry_dir)

        # Register model with good metrics
        metadata = registry.register_model(
            model_name="test_model",
            model_path=sample_model_file,
            metrics={"f1": 0.75, "accuracy": 0.73},
        )

        # Validate
        is_valid = registry.validate_model(
            model_name="test_model",
            version=metadata.version,
            min_metrics={"f1": 0.60, "accuracy": 0.60},
        )

        assert is_valid is True
        assert metadata.validation_passed is True
        assert "passed" in metadata.validation_notes.lower()

    def test_validate_model_failure(self, temp_registry_dir, sample_model_file):
        """Test failed model validation."""
        registry = ModelRegistry(registry_dir=temp_registry_dir)

        # Register model with poor metrics
        metadata = registry.register_model(
            model_name="test_model",
            model_path=sample_model_file,
            metrics={"f1": 0.50, "accuracy": 0.48},
        )

        # Validate with high threshold
        is_valid = registry.validate_model(
            model_name="test_model",
            version=metadata.version,
            min_metrics={"f1": 0.70, "accuracy": 0.70},
        )

        assert is_valid is False
        assert metadata.validation_passed is False
        assert metadata.status == ModelStatus.FAILED

    def test_promote_to_production(self, temp_registry_dir, sample_model_file):
        """Test promoting model to production."""
        registry = ModelRegistry(registry_dir=temp_registry_dir)

        # Register and validate model
        metadata = registry.register_model(
            model_name="test_model",
            model_path=sample_model_file,
            metrics={"f1": 0.75},
        )

        registry.validate_model(
            model_name="test_model",
            version=metadata.version,
            min_metrics={"f1": 0.60},
        )

        # Promote to production
        success = registry.promote_to_production(
            model_name="test_model",
            version=metadata.version,
            notes="Initial deployment",
        )

        assert success is True
        assert metadata.status == ModelStatus.PRODUCTION
        assert metadata.deployed_at is not None
        assert metadata.deployment_notes == "Initial deployment"

    def test_promote_without_validation(self, temp_registry_dir, sample_model_file):
        """Test promoting unvalidated model fails."""
        registry = ModelRegistry(registry_dir=temp_registry_dir)

        # Register but don't validate
        metadata = registry.register_model(
            model_name="test_model",
            model_path=sample_model_file,
        )

        # Try to promote
        success = registry.promote_to_production(
            model_name="test_model",
            version=metadata.version,
        )

        # Should fail
        assert success is False
        assert metadata.status == ModelStatus.STAGED

    def test_promote_demotes_previous_production(self, temp_registry_dir, sample_model_file):
        """Test that promoting new model archives previous production model."""
        registry = ModelRegistry(registry_dir=temp_registry_dir)

        # Register and promote v1.0
        metadata_v1 = registry.register_model(
            model_name="test_model",
            model_path=sample_model_file,
            version="v1.0",
            metrics={"f1": 0.70},
        )
        registry.validate_model("test_model", "v1.0", min_metrics={"f1": 0.60})
        registry.promote_to_production("test_model", "v1.0")

        # Register and promote v2.0
        metadata_v2 = registry.register_model(
            model_name="test_model",
            model_path=sample_model_file,
            version="v2.0",
            metrics={"f1": 0.75},
        )
        registry.validate_model("test_model", "v2.0", min_metrics={"f1": 0.60})
        registry.promote_to_production("test_model", "v2.0")

        # Check v1.0 is archived
        assert metadata_v1.status == ModelStatus.ARCHIVED
        # Check v2.0 is production
        assert metadata_v2.status == ModelStatus.PRODUCTION

    def test_get_production_model(self, temp_registry_dir, sample_model_file):
        """Test getting production model."""
        registry = ModelRegistry(registry_dir=temp_registry_dir)

        # Register and promote model
        metadata = registry.register_model(
            model_name="test_model",
            model_path=sample_model_file,
            version="v1.0",
            metrics={"f1": 0.75},
        )
        registry.validate_model("test_model", "v1.0", min_metrics={"f1": 0.60})
        registry.promote_to_production("test_model", "v1.0")

        # Get production model
        prod_model = registry.get_production_model("test_model")

        assert prod_model is not None
        assert prod_model.version == "v1.0"
        assert prod_model.status == ModelStatus.PRODUCTION

    def test_get_all_versions(self, temp_registry_dir, sample_model_file):
        """Test getting all model versions."""
        registry = ModelRegistry(registry_dir=temp_registry_dir)

        # Register multiple versions
        registry.register_model("test_model", sample_model_file, version="v1.0")
        registry.register_model("test_model", sample_model_file, version="v2.0")
        registry.register_model("test_model", sample_model_file, version="v3.0")

        # Get all versions
        versions = registry.get_all_versions("test_model")

        assert len(versions) == 3
        # Should be sorted by date (newest first)
        assert versions[0].version in ["v3.0", "v2.0", "v1.0"]

    def test_get_model_history(self, temp_registry_dir, sample_model_file):
        """Test getting model history."""
        registry = ModelRegistry(registry_dir=temp_registry_dir)

        # Register and promote model
        metadata = registry.register_model(
            model_name="test_model",
            model_path=sample_model_file,
            version="v1.0",
            metrics={"f1": 0.75},
        )
        registry.validate_model("test_model", "v1.0", min_metrics={"f1": 0.60})
        registry.promote_to_production("test_model", "v1.0")

        # Get history
        history = registry.get_model_history("test_model")

        assert history["model_name"] == "test_model"
        assert history["total_versions"] == 1
        assert history["production_version"] == "v1.0"
        assert len(history["versions"]) == 1

    def test_rollback_to_version(self, temp_registry_dir, sample_model_file):
        """Test rolling back to previous version."""
        registry = ModelRegistry(registry_dir=temp_registry_dir)

        # Register two versions
        metadata_v1 = registry.register_model(
            "test_model", sample_model_file, version="v1.0", metrics={"f1": 0.70}
        )
        registry.validate_model("test_model", "v1.0", min_metrics={"f1": 0.60})
        registry.promote_to_production("test_model", "v1.0")

        metadata_v2 = registry.register_model(
            "test_model", sample_model_file, version="v2.0", metrics={"f1": 0.65}
        )
        registry.validate_model("test_model", "v2.0", min_metrics={"f1": 0.60})
        registry.promote_to_production("test_model", "v2.0")

        # Rollback to v1.0
        success = registry.rollback_to_version("test_model", "v1.0")

        assert success is True
        assert metadata_v1.status == ModelStatus.PRODUCTION
        assert metadata_v2.status == ModelStatus.ARCHIVED

    def test_archive_model(self, temp_registry_dir, sample_model_file):
        """Test archiving model."""
        registry = ModelRegistry(registry_dir=temp_registry_dir)

        metadata = registry.register_model(
            "test_model", sample_model_file, version="v1.0"
        )

        # Archive model
        registry.archive_model("test_model", "v1.0")

        assert metadata.status == ModelStatus.ARCHIVED

    def test_archive_production_model_fails(self, temp_registry_dir, sample_model_file):
        """Test that archiving production model is prevented."""
        registry = ModelRegistry(registry_dir=temp_registry_dir)

        # Register and promote
        metadata = registry.register_model(
            "test_model", sample_model_file, version="v1.0", metrics={"f1": 0.75}
        )
        registry.validate_model("test_model", "v1.0", min_metrics={"f1": 0.60})
        registry.promote_to_production("test_model", "v1.0")

        # Try to archive production model
        registry.archive_model("test_model", "v1.0")

        # Should still be production
        assert metadata.status == ModelStatus.PRODUCTION

    def test_delete_model(self, temp_registry_dir, sample_model_file):
        """Test deleting model."""
        registry = ModelRegistry(registry_dir=temp_registry_dir)

        registry.register_model("test_model", sample_model_file, version="v1.0")

        # Delete model
        registry.delete_model("test_model", "v1.0", delete_files=False)

        # Should be removed from registry
        versions = registry.get_all_versions("test_model")
        assert len(versions) == 0

    def test_persistence(self, temp_registry_dir, sample_model_file):
        """Test registry persistence across instances."""
        # Create registry and register model
        registry1 = ModelRegistry(registry_dir=temp_registry_dir)
        registry1.register_model(
            "test_model", sample_model_file, version="v1.0", metrics={"f1": 0.75}
        )

        # Create new registry instance
        registry2 = ModelRegistry(registry_dir=temp_registry_dir)

        # Should load previous data
        versions = registry2.get_all_versions("test_model")
        assert len(versions) == 1
        assert versions[0].version == "v1.0"
        assert versions[0].metrics["f1"] == 0.75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
