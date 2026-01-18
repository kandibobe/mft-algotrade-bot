"""
Unit Tests for ML Training Pipeline
===================================

Tests the end-to-end ML pipeline with mocked data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml.pipeline import MLTrainingPipeline
from src.ml.training.model_registry import ModelRegistry


@pytest.fixture
def mock_data_dir(tmp_path):
    """Create temporary data directory with dummy feather file."""
    data_dir = tmp_path / "data" / "binance"
    data_dir.mkdir(parents=True)

    # Create dummy OHLCV with clear trends to ensure multi-class labels
    dates = pd.date_range(start="2024-01-01", periods=1000, freq="5min")

    # Sine wave trend
    t = np.linspace(0, 4*np.pi, 1000)
    trend = 50000 + 1000 * np.sin(t)
    noise = np.random.randn(1000) * 50

    df = pd.DataFrame({
        "open": trend + noise,
        "high": trend + noise + 100,
        "low": trend + noise - 100,
        "close": trend + noise,
        "volume": np.random.randint(100, 1000, 1000),
        "date": dates
    })

    # Save as feather
    df.to_feather(data_dir / "BTC_USDT-5m.feather")

    return data_dir

@pytest.fixture
def mock_models_dir(tmp_path):
    """Create temporary models directory."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return models_dir

def test_pipeline_end_to_end(mock_data_dir, mock_models_dir):
    """Test full training cycle."""

    # Initialize pipeline
    # Override paths to use temp dirs
    pipeline = MLTrainingPipeline(
        data_dir=str(mock_data_dir),
        models_dir=str(mock_models_dir),
        quick_mode=True
    )

    # Run pipeline
    results = pipeline.run(
        pairs=["BTC/USDT"],
        timeframe="5m",
        optimize=False, # Skip optuna for speed
        n_trials=1
    )

    # Check results
    assert "BTC/USDT" in results
    assert results["BTC/USDT"]["success"] is True

    # Check registry
    registry = ModelRegistry(registry_dir=str(mock_models_dir / "registry"))
    print(f"Registry models: {registry.models.keys()}")
    models = registry.get_all_versions("BTC_USDT")
    assert len(models) > 0
    assert models[0].status.value == "staged"

    # Check file exists
    assert Path(models[0].model_path).exists()
