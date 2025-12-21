"""
Tests for Online Learning (Model Retraining) Module.
"""

import numpy as np
import pytest
import tempfile
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression

from src.ml.online_learner import (
    OnlineLearner,
    OnlineLearningConfig,
    load_model,
    save_model,
    RIVER_AVAILABLE
)


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    return X, y


@pytest.fixture
def dummy_model():
    """Create a dummy model for testing."""
    model = LogisticRegression(random_state=42)
    X = np.random.randn(10, 5)
    y = np.random.randint(0, 2, 10)
    model.fit(X, y)
    return model


@pytest.fixture
def model_path(dummy_model):
    """Create a temporary model file."""
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        pickle.dump(dummy_model, f)
        return f.name


def test_load_save_model(dummy_model):
    """Test model loading and saving."""
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        model_path = f.name
    
    try:
        # Save model
        assert save_model(dummy_model, model_path)
        
        # Load model
        loaded_model = load_model(model_path)
        assert loaded_model is not None
        
        # Verify predictions match
        X_test = np.random.randn(5, 5)
        original_preds = dummy_model.predict(X_test)
        loaded_preds = loaded_model.predict(X_test)
        assert np.array_equal(original_preds, loaded_preds)
        
    finally:
        Path(model_path).unlink(missing_ok=True)


def test_online_learner_initialization(model_path):
    """Test OnlineLearner initialization."""
    # Test with existing model
    learner = OnlineLearner(model_path)
    assert learner.prod_model is not None
    assert learner.online_model is not None
    assert learner.config is not None
    assert learner.update_count == 0
    
    # Test with non-existent model (should create dummy)
    learner2 = OnlineLearner("non_existent_model.pkl")
    assert learner2.prod_model is not None
    assert learner2.online_model is not None


def test_online_learner_config():
    """Test OnlineLearningConfig."""
    config = OnlineLearningConfig(
        base_model_path="test.pkl",
        learning_rate=0.1,
        improvement_threshold=0.1,
        ab_test_traffic_pct=0.2
    )
    
    assert config.base_model_path == "test.pkl"
    assert config.learning_rate == 0.1
    assert config.improvement_threshold == 0.1
    assert config.ab_test_traffic_pct == 0.2
    # use_river should match RIVER_AVAILABLE
    assert config.use_river == RIVER_AVAILABLE


def test_update_online(sample_data):
    """Test online model updates."""
    X, y = sample_data
    
    # Create learner with dummy model
    learner = OnlineLearner("dummy.pkl")
    
    # Update with single sample
    initial_count = learner.update_count
    learner.update_online(X[0], y[0])
    
    assert learner.update_count == initial_count + 1
    assert len(learner.prod_performance_history) == 1
    assert len(learner.online_performance_history) == 1
    
    # Update with multiple samples
    for i in range(1, 10):
        learner.update_online(X[i], y[i])
    
    assert learner.update_count == initial_count + 10


def test_predict_methods(sample_data):
    """Test prediction methods."""
    X, y = sample_data
    learner = OnlineLearner("dummy.pkl")
    
    # Test single prediction
    pred = learner.predict(X[0], use_ab_test=False)
    assert pred in [0, 1]
    
    # Test gradual rollout (A/B testing)
    pred_ab = learner.gradual_rollout(X[0])
    assert pred_ab in [0, 1]
    
    # Test with custom traffic percentage
    pred_custom = learner.gradual_rollout(X[0], traffic_pct=0.5)
    assert pred_custom in [0, 1]


def test_ab_test_management():
    """Test A/B test start/stop functionality."""
    learner = OnlineLearner("dummy.pkl")
    
    # Start A/B test
    assert learner.start_ab_test(traffic_pct=0.3)
    assert learner.ab_test_active
    assert learner.config.ab_test_traffic_pct == 0.3
    assert 'start_time' in learner.ab_test_results
    
    # Stop A/B test
    results = learner.stop_ab_test()
    assert not learner.ab_test_active
    assert 'end_time' in results
    assert 'duration' in results
    assert 'total_samples' in results
    
    # Try to stop non-active test
    results = learner.stop_ab_test()
    assert 'error' in results


def test_model_replacement_logic(sample_data):
    """Test model replacement decision logic."""
    X, y = sample_data
    learner = OnlineLearner("dummy.pkl")
    
    # Initially should not replace (not enough samples)
    assert not learner.should_replace_prod_model()
    
    # Update with many samples to simulate improvement
    # Note: In real scenario, online model might improve with updates
    for i in range(learner.config.min_samples_for_comparison):
        learner.update_online(X[i % len(X)], y[i % len(y)])
    
    # Check replacement logic (may or may not trigger depending on performance)
    should_replace = learner.should_replace_prod_model()
    assert isinstance(should_replace, bool)


def test_performance_stats():
    """Test performance statistics retrieval."""
    learner = OnlineLearner("dummy.pkl")
    
    stats = learner.get_performance_stats()
    
    assert 'production_model' in stats
    assert 'online_model' in stats
    assert 'comparison' in stats
    
    prod_stats = stats['production_model']
    assert 'accuracy' in prod_stats
    assert 'update_count' in prod_stats
    assert 'performance_history' in prod_stats
    assert 'avg_performance' in prod_stats
    
    online_stats = stats['online_model']
    assert 'accuracy' in online_stats
    assert 'update_count' in online_stats
    assert 'performance_history' in online_stats
    assert 'avg_performance' in online_stats
    
    comp_stats = stats['comparison']
    assert 'improvement' in comp_stats
    assert 'should_replace' in comp_stats
    assert 'drift_detected' in comp_stats


def test_batch_operations(sample_data):
    """Test batch update and evaluation."""
    X, y = sample_data
    learner = OnlineLearner("dummy.pkl")
    
    # Test batch update
    X_batch = X[:10]
    y_batch = y[:10]
    
    initial_count = learner.update_count
    learner.batch_update(X_batch, y_batch)
    assert learner.update_count == initial_count + len(X_batch)
    
    # Test batch evaluation
    eval_results = learner.evaluate_on_batch(X_batch, y_batch)
    
    assert 'prod_accuracy' in eval_results
    assert 'online_accuracy' in eval_results
    assert 'total_samples' in eval_results
    assert 'prod_correct' in eval_results
    assert 'online_correct' in eval_results
    
    assert 0 <= eval_results['prod_accuracy'] <= 1
    assert 0 <= eval_results['online_accuracy'] <= 1
    assert eval_results['total_samples'] == len(X_batch)


def test_model_reset():
    """Test online model reset functionality."""
    learner = OnlineLearner("dummy.pkl")
    
    # Update model to change its state
    X = np.random.randn(5)
    y = 1
    learner.update_online(X, y)
    
    # Reset model
    learner.reset_online_model()
    
    # Verify reset
    assert learner.update_count > 0  # update_count should not reset
    # Online model should be reinitialized


def test_drift_detection_initialization():
    """Test drift detector initialization."""
    # Test with drift detection enabled
    config = OnlineLearningConfig(enable_drift_detection=True)
    learner = OnlineLearner("dummy.pkl", config=config)
    assert learner.drift_detector is not None
    
    # Test with drift detection disabled
    config = OnlineLearningConfig(enable_drift_detection=False)
    learner = OnlineLearner("dummy.pkl", config=config)
    assert learner.drift_detector is None


if __name__ == "__main__":
    # Run tests
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
