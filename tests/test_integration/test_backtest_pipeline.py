"""
Integration tests for backtest pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def sample_backtest_data():
    """Generate sample data for backtest testing."""
    np.random.seed(42)
    n = 500
    
    # Generate realistic price data
    returns = np.random.randn(n) * 0.02
    close = 100 * np.exp(np.cumsum(returns))
    
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    
    dates = pd.date_range('2024-01-01', periods=n, freq='5min')
    
    return pd.DataFrame({
        'date': dates,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


@pytest.mark.integration
class TestDataPipeline:
    """Integration tests for data loading pipeline."""
    
    def test_load_sample_data(self):
        """Test loading sample CSV data."""
        from src.data.loader import load_csv
        
        fixture_path = Path('tests/fixtures/sample_data/BTC_USDT-5m.csv')
        
        if fixture_path.exists():
            df = load_csv(fixture_path)
            
            assert df is not None
            assert len(df) > 0
            assert 'open' in df.columns
            assert 'close' in df.columns
    
    def test_validate_sample_data(self):
        """Test data validation on sample data."""
        from src.data.loader import load_csv
        from src.data.validator import validate_ohlcv
        
        fixture_path = Path('tests/fixtures/sample_data/BTC_USDT-5m.csv')
        
        if fixture_path.exists():
            df = load_csv(fixture_path)
            is_valid, issues = validate_ohlcv(df)
            
            assert is_valid is True
            assert len(issues) == 0


@pytest.mark.integration
class TestIndicatorPipeline:
    """Integration tests for indicator calculation."""
    
    def test_calculate_all_indicators(self, sample_backtest_data):
        """Test full indicator calculation pipeline."""
        from src.utils.indicators import calculate_all_indicators
        
        result = calculate_all_indicators(sample_backtest_data)
        
        # Check key indicators exist and have values
        assert 'ema_50' in result.columns
        assert 'rsi' in result.columns
        assert 'macd' in result.columns
        assert 'atr' in result.columns
        
        # Check no columns are all NaN (after warmup)
        warmup = 200
        for col in ['ema_50', 'rsi', 'macd', 'atr']:
            assert not result[col].iloc[warmup:].isna().all()
    
    def test_regime_detection(self, sample_backtest_data):
        """Test regime detection pipeline."""
        from src.utils.regime_detection import calculate_regime_score
        
        df = sample_backtest_data
        
        regime_data = calculate_regime_score(
            df['high'], df['low'], df['close'], df['volume']
        )
        
        assert 'regime_score' in regime_data.columns
        assert 'risk_factor' in regime_data.columns
        
        # Regime score should be 0-100
        valid_scores = regime_data['regime_score'].dropna()
        assert valid_scores.min() >= 0
        assert valid_scores.max() <= 100


@pytest.mark.integration
class TestRiskPipeline:
    """Integration tests for risk calculations."""
    
    def test_risk_metrics_calculation(self, sample_backtest_data):
        """Test risk metrics on simulated equity curve."""
        from src.utils.risk import calculate_risk_metrics
        
        # Create equity curve from returns
        returns = sample_backtest_data['close'].pct_change().dropna()
        equity = (1 + returns).cumprod() * 10000
        
        metrics = calculate_risk_metrics(equity)
        
        assert 'total_return' in metrics
        assert 'max_drawdown' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'sortino_ratio' in metrics
        
        # Max drawdown should be positive (it's the magnitude)
        assert metrics['max_drawdown'] >= 0
    
    def test_position_sizing(self):
        """Test position sizing calculation."""
        from src.utils.risk import calculate_position_size_fixed_risk
        
        position = calculate_position_size_fixed_risk(
            account_balance=10000,
            risk_per_trade=0.02,  # 2%
            entry_price=50000,
            stop_loss_price=47500,  # 5% below
            min_position=0.001,
            max_position=1.0
        )
        
        # Should risk $200 (2% of 10000)
        # Stop distance is $2500, so position = 200/2500 = 0.08 BTC
        assert 0.05 < position < 0.15


@pytest.mark.integration
class TestStrategyPipeline:
    """Integration tests for strategy execution."""
    
    def test_strategy_config_loading(self):
        """Test strategy config from YAML."""
        from src.strategies.strategy_config import StrategyConfig
        
        config_path = Path('config/strategy_config.yaml')
        
        if config_path.exists():
            config = StrategyConfig.from_file(str(config_path))
            
            assert config.name == 'StoicEnsembleStrategy'
            assert 0 < config.risk_per_trade < 0.1
            assert config.stoploss < 0
    
    def test_strategy_config_validation(self):
        """Test strategy config validation."""
        from src.strategies.strategy_config import StrategyConfig
        
        # Valid config
        config = StrategyConfig(
            risk_per_trade=0.02,
            max_positions=3,
            stoploss=-0.05
        )
        
        assert config.validate() is True
        
        # Invalid config
        with pytest.raises(ValueError):
            invalid = StrategyConfig(
                risk_per_trade=0.5,  # Too high
                stoploss=0.1  # Should be negative
            )
            invalid.validate()


@pytest.mark.integration
class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_indicator_to_signal_pipeline(self, sample_backtest_data):
        """Test full pipeline from data to signals."""
        from src.utils.indicators import calculate_all_indicators
        from src.utils.regime_detection import calculate_regime_score, get_regime_parameters
        
        # Step 1: Calculate indicators
        df = calculate_all_indicators(sample_backtest_data)
        
        # Step 2: Calculate regime
        regime_data = calculate_regime_score(
            df['high'], df['low'], df['close'], df['volume']
        )
        
        # Step 3: Get parameters
        current_score = regime_data['regime_score'].iloc[-1]
        params = get_regime_parameters(current_score)
        
        # Verify complete pipeline
        assert 'ema_50' in df.columns
        assert 'regime_score' in regime_data.columns
        assert 'mode' in params
        assert params['risk_per_trade'] > 0
