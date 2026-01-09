"""
Integration Tests for Position Sizing Flow
==========================================

Tests the interaction between the Strategy, Mixin, and RiskManager
specifically for the custom_stake_amount (sizing) logic.
Ensures that inheritance works and parent limits are respected.
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from datetime import datetime
from src.strategies.risk_mixin import StoicRiskMixin
from src.risk.risk_manager import RiskManager

class MockStrategy(StoicRiskMixin):
    """Test strategy inheriting from Mixin."""
    def __init__(self):
        self.config = {'dry_run': True, 'runmode': 'live'}
        self.timeframe = '5m'
        self.stoploss = -0.10
        self.wallets = MagicMock()
        self.dp = MagicMock()
        super().__init__()

@pytest.fixture
def sizing_strategy():
    strat = MockStrategy()
    
    # Mock Config
    mock_conf = MagicMock()
    mock_conf.risk.max_drawdown_pct = 0.15
    mock_conf.risk.max_daily_loss_pct = 0.05
    mock_conf.risk.max_position_pct = 0.20
    mock_conf.risk.max_portfolio_risk = 0.10
    mock_conf.risk.max_correlation = 0.85
    mock_conf.risk.liquidation_buffer = 0.05
    mock_conf.risk.max_safe_leverage = 5.0
    mock_conf.dry_run = True
    mock_conf.validate_for_live_trading.return_value = []
    
    with patch('src.strategies.risk_mixin.ConfigurationManager.get_config', return_value=mock_conf):
        strat.bot_start()
        
    return strat

def test_sizing_respects_volatility(sizing_strategy):
    """Test that sizing varies with ATR (Inverse Volatility)."""
    pair = "BTC/USDT"
    sizing_strategy.wallets.get_total_stake_amount.return_value = 10000.0
    
    # Low Volatility Case (ATR = 1%)
    # Sizing: (10000 * 0.01) / 0.01 = 10000. Capped at 20% = 2000.
    df_low = pd.DataFrame({
        'date': [datetime.now()],
        'close': [50000.0],
        'atr_percent': [0.01]
    })
    sizing_strategy.dp.get_analyzed_dataframe.return_value = (df_low, None)
    
    size_low = sizing_strategy.custom_stake_amount(
        pair, datetime.now(), 50000.0, 1000.0, 10.0, 5000.0, 1.0, None, "long"
    )
    
    # High Volatility Case (ATR = 10% - high enough to NOT be capped)
    # Sizing: (10000 * 0.01) / 0.10 = 1000.
    df_high = pd.DataFrame({
        'date': [datetime.now()],
        'close': [50000.0],
        'atr_percent': [0.10]
    })
    sizing_strategy.dp.get_analyzed_dataframe.return_value = (df_high, None)
    
    size_high = sizing_strategy.custom_stake_amount(
        pair, datetime.now(), 50000.0, 1000.0, 10.0, 5000.0, 1.0, None, "long"
    )
    
    # Higher volatility should mean smaller size
    assert size_high < size_low
    assert size_low == 2000.0 
    assert size_high == 1000.0

def test_drawdown_suppression(sizing_strategy):
    """Test that sizing is reduced during drawdown."""
    pair = "ETH/USDT"
    sizing_strategy.wallets.get_total_stake_amount.return_value = 10000.0
    
    # Use high ATR so we are not capped by the 20% max position limit
    # (10000 * 0.01) / 0.05 = 2000. (20% is 2000).
    df = pd.DataFrame({
        'date': [datetime.now()],
        'close': [3000.0],
        'atr_percent': [0.05]
    })
    sizing_strategy.dp.get_analyzed_dataframe.return_value = (df, None)
    
    # Normal State (0% DD)
    metrics_normal = MagicMock()
    metrics_normal.current_drawdown_pct = 0.0
    sizing_strategy.risk_manager.get_metrics = MagicMock(return_value=metrics_normal)
    
    size_normal = sizing_strategy.custom_stake_amount(
        pair, datetime.now(), 3000.0, 1000.0, 10.0, 5000.0, 1.0, None, "long"
    )
    
    # Drawdown State (6% DD > 5% trigger)
    metrics_dd = MagicMock()
    metrics_dd.current_drawdown_pct = 0.06
    sizing_strategy.risk_manager.get_metrics = MagicMock(return_value=metrics_dd)
    
    size_dd = sizing_strategy.custom_stake_amount(
        pair, datetime.now(), 3000.0, 1000.0, 10.0, 5000.0, 1.0, None, "long"
    )
    
    # DD size should be 50% of normal size
    assert size_dd == pytest.approx(size_normal * 0.5)
