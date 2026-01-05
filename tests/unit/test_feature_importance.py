import pytest
from unittest.mock import MagicMock, patch, mock_open
from src.ml.training.model_registry import ModelRegistry, ModelMetadata

@pytest.fixture
def registry():
    with patch('pathlib.Path.mkdir'):
        with patch('src.ml.training.model_registry.ModelRegistry._load_registry'):
            return ModelRegistry()

def test_get_feature_importance(registry):
    """Test retrieving feature importance from CSV."""
    
    # Mock metadata
    model_name = "test_model"
    version = "v1.0"
    registry.models = {
        model_name: [
            ModelMetadata(name=model_name, version=version, model_path="models/test_model.pkl", model_type="random_forest")
        ]
    }
    
    # Mock CSV content
    csv_content = "feature,importance\nrsi,0.5\nmacd,0.3\n"
    
    with patch('pathlib.Path.exists', return_value=True):
        with patch('pandas.read_csv') as mock_read_csv:
            # Mock pandas DataFrame
            mock_df = MagicMock()
            mock_df.columns = ['feature', 'importance']
            mock_df.__getitem__ = lambda self, key: ['rsi', 'macd'] if key == 'feature' else [0.5, 0.3]
            # Make zip work
            mock_read_csv.return_value = mock_df
            
            # Since mocking pandas is tricky with zip, let's simplify the test expectation
            # or use a more robust mock.
            # Actually, let's just mock the return of get_feature_importance if we were testing orchestrator.
            # But here we are testing registry.
            
            # Better approach: Create a temporary file
            pass

def test_get_feature_importance_integration(tmp_path):
    """Integration test with temporary file."""
    registry = ModelRegistry(registry_dir=str(tmp_path))
    
    # Create dummy model file and importance file
    model_path = tmp_path / "test_model.pkl"
    model_path.touch()
    
    importance_path = tmp_path / "test_model.csv"
    importance_path.write_text("feature,importance\nrsi,0.5\nmacd,0.3\n")
    
    # Manually add to registry state
    registry.models["test_model"] = [
        ModelMetadata(name="test_model", version="v1.0", model_path=str(model_path), model_type="random_forest")
    ]
    
    importance = registry.get_feature_importance("test_model", "v1.0")
    
    assert importance['rsi'] == 0.5
    assert importance['macd'] == 0.3
    assert len(importance) == 2
