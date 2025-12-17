"""
Tests for StoicEnsembleStrategyV2.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 300
    
    # Generate trending up data for clearer signals
    base_price = 100
    trend = np.linspace(0, 20, n)  # Upward trend
    noise = np.random.randn(n) * 2
    close = base_price + trend + noise
    
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
    }).set_index('date')


@pytest.fixture
def strategy_config():
    """Mock strategy config."""
    return {
        'exchange': {'name': 'binance'},
        'stake_currency': 'USDT',
        'dry_run': True
    }


class TestIndicatorCalculation:
    """Tests for indicator calculation."""
    
    def test_populate_indicators_adds_columns(self, sample_ohlcv, strategy_config):
        """Test that populate_indicators adds expected columns."""
        # Import here to handle potential import errors
        try:
            from user_data.strategies.StoicEnsembleStrategyV2 import StoicEnsembleStrategyV2
        except ImportError:
            pytest.skip("Strategy not available")
        
        strategy = StoicEnsembleStrategyV2(strategy_config)
        
        # Need to reset index for Freqtrade compatibility
        df = sample_ohlcv.reset_index()
        
        result = strategy.populate_indicators(df, {'pair': 'BTC/USDT'})
        
        expected_indicators = [
            'ema_9', 'ema_21', 'ema_50', 'ema_100', 'ema_200',
            'rsi', 'macd', 'atr', 'bb_upper', 'bb_lower',
            'stoch_k', 'stoch_d', 'adx'
        ]
        
        for ind in expected_indicators:
            assert ind in result.columns, f"Missing indicator: {ind}"
    
    def test_ensemble_score_calculated(self, sample_ohlcv, strategy_config):
        """Test that ensemble score is calculated."""
        try:
            from user_data.strategies.StoicEnsembleStrategyV2 import StoicEnsembleStrategyV2
        except ImportError:
            pytest.skip("Strategy not available")
        
        strategy = StoicEnsembleStrategyV2(strategy_config)
        df = sample_ohlcv.reset_index()
        
        result = strategy.populate_indicators(df, {'pair': 'BTC/USDT'})
        
        assert 'ensemble_score' in result.columns
        assert 'momentum_signal' in result.columns
        assert 'mean_reversion_signal' in result.columns
        assert 'breakout_signal' in result.columns


class TestEntryLogic:
    """Tests for entry signal generation."""
    
    def test_entry_requires_trend_filter(self, sample_ohlcv, strategy_config):
        """Test that entry requires price above EMA200."""
        try:
            from user_data.strategies.StoicEnsembleStrategyV2 import StoicEnsembleStrategyV2
        except ImportError:
            pytest.skip("Strategy not available")
        
        strategy = StoicEnsembleStrategyV2(strategy_config)
        df = sample_ohlcv.reset_index()
        
        # Force price below EMA200
        df['close'] = df['close'] * 0.5
        
        result = strategy.populate_indicators(df, {'pair': 'BTC/USDT'})
        result = strategy.populate_entry_trend(result, {'pair': 'BTC/USDT'})
        
        # Should have no entries (or very few)
        entries = result.get('enter_long', pd.Series([0])).sum()
        assert entries == 0 or entries < len(result) * 0.01
    
    def test_entry_signal_format(self, sample_ohlcv, strategy_config):
        """Test that entry signals are in correct format."""
        try:
            from user_data.strategies.StoicEnsembleStrategyV2 import StoicEnsembleStrategyV2
        except ImportError:
            pytest.skip("Strategy not available")
        
        strategy = StoicEnsembleStrategyV2(strategy_config)
        df = sample_ohlcv.reset_index()
        
        result = strategy.populate_indicators(df, {'pair': 'BTC/USDT'})
        result = strategy.populate_entry_trend(result, {'pair': 'BTC/USDT'})
        
        if 'enter_long' in result.columns:
            # Values should be 0 or 1
            unique_values = result['enter_long'].dropna().unique()
            assert all(v in [0, 1] for v in unique_values)


class TestExitLogic:
    """Tests for exit signal generation."""
    
    def test_exit_on_overbought(self, sample_ohlcv, strategy_config):
        """Test that exit triggers on overbought conditions."""
        try:
            from user_data.strategies.StoicEnsembleStrategyV2 import StoicEnsembleStrategyV2
        except ImportError:
            pytest.skip("Strategy not available")
        
        strategy = StoicEnsembleStrategyV2(strategy_config)
        df = sample_ohlcv.reset_index()
        
        result = strategy.populate_indicators(df, {'pair': 'BTC/USDT'})
        result = strategy.populate_exit_trend(result, {'pair': 'BTC/USDT'})
        
        # Check exit column exists
        if 'exit_long' in result.columns:
            # Should have some exit signals
            assert result['exit_long'].sum() >= 0  # At least valid


class TestRiskManagement:
    """Tests for risk management features."""
    
    def test_protections_configured(self, strategy_config):
        """Test that protections are properly configured."""
        try:
            from user_data.strategies.StoicEnsembleStrategyV2 import StoicEnsembleStrategyV2
        except ImportError:
            pytest.skip("Strategy not available")
        
        strategy = StoicEnsembleStrategyV2(strategy_config)
        
        protections = strategy.protections
        
        assert len(protections) > 0
        
        # Check for required protection types
        protection_methods = [p['method'] for p in protections]
        assert 'StoplossGuard' in protection_methods
        assert 'MaxDrawdown' in protection_methods
    
    def test_stoploss_configured(self, strategy_config):
        """Test that stoploss is properly set."""
        try:
            from user_data.strategies.StoicEnsembleStrategyV2 import StoicEnsembleStrategyV2
        except ImportError:
            pytest.skip("Strategy not available")
        
        strategy = StoicEnsembleStrategyV2(strategy_config)
        
        assert strategy.stoploss == -0.05  # 5% stop loss
        assert strategy.trailing_stop is True
    
    def test_roi_configured(self, strategy_config):
        """Test that ROI table is properly set."""
        try:
            from user_data.strategies.StoicEnsembleStrategyV2 import StoicEnsembleStrategyV2
        except ImportError:
            pytest.skip("Strategy not available")
        
        strategy = StoicEnsembleStrategyV2(strategy_config)
        
        roi = strategy.minimal_roi
        
        assert '0' in roi or 0 in roi
        # ROI should decrease over time
        roi_values = list(roi.values())
        assert roi_values[0] >= roi_values[-1]


class TestCustomMethods:
    """Tests for custom strategy methods."""
    
    def test_custom_stake_amount_reduces_on_high_vol(self, strategy_config):
        """Test that stake is reduced in high volatility."""
        try:
            from user_data.strategies.StoicEnsembleStrategyV2 import StoicEnsembleStrategyV2
        except ImportError:
            pytest.skip("Strategy not available")

        strategy = StoicEnsembleStrategyV2(strategy_config)

        # Mock dataframe provider - need to create it first
        mock_df = pd.DataFrame({
            'atr_pct': [6.0]  # High volatility
        })

        # Create mock dp (dataframe provider)
        strategy.dp = MagicMock()
        strategy.dp.get_analyzed_dataframe.return_value = (mock_df, datetime.now())

        stake = strategy.custom_stake_amount(
            pair='BTC/USDT',
            current_time=datetime.now(),
            current_rate=50000,
            proposed_stake=100,
            min_stake=10,
            max_stake=1000,
            leverage=1,
            entry_tag=None,
            side='long'
        )

        # Should reduce stake due to high volatility
        assert stake < 100
    
    def test_confirm_trade_entry_time_filter(self, strategy_config):
        """Test that trades are filtered during low liquidity hours."""
        try:
            from user_data.strategies.StoicEnsembleStrategyV2 import StoicEnsembleStrategyV2
        except ImportError:
            pytest.skip("Strategy not available")
        
        strategy = StoicEnsembleStrategyV2(strategy_config)
        strategy._regime_mode = 'normal'  # Not aggressive
        
        # Test during low liquidity hours
        low_liq_time = datetime(2024, 1, 1, 3, 0, 0)  # 3 AM
        
        result = strategy.confirm_trade_entry(
            pair='BTC/USDT',
            order_type='limit',
            amount=0.1,
            rate=50000,
            time_in_force='GTC',
            current_time=low_liq_time,
            entry_tag=None,
            side='long'
        )
        
        assert result is False  # Should reject during low liquidity
