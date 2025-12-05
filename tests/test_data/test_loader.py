"""
Tests for data loader module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.loader import get_ohlcv, load_csv, get_data_hash, get_data_metadata
from src.data.validator import validate_ohlcv, check_data_integrity


# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent.parent / 'fixtures' / 'sample_data'


class TestDataLoader:
    """Tests for data loading functionality."""
    
    def test_load_csv_basic(self):
        """Test basic CSV loading."""
        csv_path = FIXTURES_DIR / 'BTC_USDT-5m.csv'
        df = load_csv(csv_path)
        
        assert df is not None
        assert len(df) > 0
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns
        assert 'volume' in df.columns
    
    def test_load_csv_has_correct_types(self):
        """Test that loaded data has correct types."""
        csv_path = FIXTURES_DIR / 'BTC_USDT-5m.csv'
        df = load_csv(csv_path)
        
        # Price columns should be numeric
        assert pd.api.types.is_numeric_dtype(df['open'])
        assert pd.api.types.is_numeric_dtype(df['high'])
        assert pd.api.types.is_numeric_dtype(df['low'])
        assert pd.api.types.is_numeric_dtype(df['close'])
        assert pd.api.types.is_numeric_dtype(df['volume'])
    
    def test_data_hash_deterministic(self):
        """Test that data hash is deterministic."""
        csv_path = FIXTURES_DIR / 'BTC_USDT-5m.csv'
        df = load_csv(csv_path)
        
        hash1 = get_data_hash(df)
        hash2 = get_data_hash(df)
        
        assert hash1 == hash2
        assert len(hash1) == 12  # Expected hash length
    
    def test_data_hash_changes_with_data(self):
        """Test that hash changes when data changes."""
        # Create synthetic dataframes to ensure hash changes
        df1 = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [98.0, 99.0, 100.0],
            'close': [103.0, 104.0, 105.0],
            'volume': [1000.0, 1100.0, 1200.0]
        })
        
        df2 = df1.copy()
        df2.loc[0, 'close'] = 999.0  # Modify value
        
        hash1 = get_data_hash(df1)
        hash2 = get_data_hash(df2)
        
        assert hash1 != hash2, f"Hash should change with data: {hash1} vs {hash2}"
    
    def test_get_data_metadata(self):
        """Test metadata generation."""
        csv_path = FIXTURES_DIR / 'BTC_USDT-5m.csv'
        df = load_csv(csv_path)
        
        # Set datetime index for metadata
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        metadata = get_data_metadata(df, 'BTC/USDT', '5m')
        
        assert 'symbol' in metadata
        assert 'timeframe' in metadata
        assert 'num_candles' in metadata
        assert 'data_hash' in metadata
        assert metadata['symbol'] == 'BTC/USDT'
        assert metadata['timeframe'] == '5m'
        assert metadata['num_candles'] == len(df)


class TestDataValidator:
    """Tests for data validation functionality."""
    
    def test_validate_valid_data(self):
        """Test validation passes for valid data."""
        csv_path = FIXTURES_DIR / 'BTC_USDT-5m.csv'
        df = load_csv(csv_path)
        
        is_valid, issues = validate_ohlcv(df)
        
        assert is_valid is True
        assert len(issues) == 0
    
    def test_validate_missing_column(self):
        """Test validation fails for missing columns."""
        df = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            # Missing 'close' and 'volume'
        })
        
        is_valid, issues = validate_ohlcv(df)
        
        assert is_valid is False
        assert any('Missing columns' in issue for issue in issues)
    
    def test_validate_negative_values(self):
        """Test validation catches negative values."""
        df = pd.DataFrame({
            'open': [100, -101],  # Negative value
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102],
            'volume': [1000, 1100]
        })
        
        is_valid, issues = validate_ohlcv(df)
        
        assert is_valid is False
        assert any('Negative' in issue for issue in issues)
    
    def test_validate_price_integrity(self):
        """Test validation catches price integrity issues."""
        df = pd.DataFrame({
            'open': [100, 101],
            'high': [98, 103],  # High < Open (invalid)
            'low': [99, 100],
            'close': [101, 102],
            'volume': [1000, 1100]
        })
        
        is_valid, issues = validate_ohlcv(df)
        
        assert is_valid is False
        assert any('integrity' in issue.lower() for issue in issues)
    
    def test_check_data_integrity_complete(self):
        """Test integrity check with complete data."""
        # Create data with no gaps
        dates = pd.date_range('2024-01-01', periods=12, freq='5min')
        df = pd.DataFrame({
            'open': [100] * 12,
            'high': [101] * 12,
            'low': [99] * 12,
            'close': [100.5] * 12,
            'volume': [1000] * 12
        }, index=dates)
        
        has_gaps, info = check_data_integrity(df, '5m')
        
        assert has_gaps is False
        assert info['completeness_pct'] == 100
        assert info['missing_candles'] == 0


class TestIntegration:
    """Integration tests for data pipeline."""
    
    def test_load_validate_complete_pipeline(self):
        """Test complete pipeline: load -> validate."""
        csv_path = FIXTURES_DIR / 'BTC_USDT-5m.csv'
        
        # Load
        df = load_csv(csv_path)
        assert df is not None
        
        # Validate
        is_valid, issues = validate_ohlcv(df)
        assert is_valid is True
        
        # Generate metadata
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        metadata = get_data_metadata(df, 'BTC/USDT', '5m')
        assert metadata['num_candles'] == len(df)
