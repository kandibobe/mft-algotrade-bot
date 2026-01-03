"""
Integration Tests for Risk <-> Strategy Flow
============================================

Tests the interaction between the Strategy (via StoicRiskMixin) and the RiskManager.
Ensures that risk controls actually block trades and trigger exits.
"""

import pytest
import logging
from unittest.mock import MagicMock, patch
from datetime import datetime
from src.strategies.risk_mixin import StoicRiskMixin
from src.risk.risk_manager import RiskManager
from src.risk.circuit_breaker import TripReason, CircuitState

@pytest.mark.integration
class TestRiskStrategyFlow:
    """Test interaction between Strategy and Risk Manager."""

    @pytest.fixture
    def strategy(self):
        """Create a test strategy instance with Mixin."""
        class TestStrategy(StoicRiskMixin):
            def __init__(self):
                # Mock attributes expected by Mixin/Strategy
                self.config = {'dry_run': True}
                self.stoploss = -0.10
                self.timeframe = '5m'
                self.wallets = MagicMock()
                self.wallets.get_total_stake_amount.return_value = 10000.0
                super().__init__()
        
        strat = TestStrategy()
        # Mock load_config to prevent file system access
        with patch('src.strategies.risk_mixin.load_config') as mock_config:
            # Mock config object
            mock_conf_obj = MagicMock()
            mock_conf_obj.risk.max_drawdown_pct = 0.10
            mock_conf_obj.risk.max_daily_loss_pct = 0.05
            mock_conf_obj.validate_for_live_trading.return_value = []
            mock_config.return_value = mock_conf_obj
            
            strat.bot_start()
            
        return strat

    def test_risk_manager_initialization(self, strategy):
        """Test that RiskManager is correctly initialized."""
        assert strategy.risk_manager is not None
        assert strategy.risk_manager.circuit_breaker is not None
        assert strategy.risk_manager.position_sizer is not None

    def test_circuit_breaker_rejects_trade(self, strategy):
        """Test that RiskManager rejects trades when Circuit Breaker is tripped."""
        # Force Trip Circuit Breaker
        strategy.risk_manager.circuit_breaker.manual_stop()
        
        # Attempt Trade
        allowed = strategy.confirm_trade_entry(
            pair="BTC/USDT",
            order_type="market",
            amount=1.0,
            rate=50000.0,
            time_in_force="gtc",
            current_time=datetime.now(),
            entry_tag="test",
            side="long"
        )
        
        assert allowed is False, "Trade should be rejected when Circuit Breaker is open"
        assert strategy.risk_manager.circuit_breaker.state == CircuitState.OPEN

    def test_emergency_exit_signal(self, strategy):
        """Test that emergency exit signal is propagated via custom_exit."""
        # Trigger Emergency
        strategy.risk_manager.emergency_stop()
        
        assert strategy.risk_manager.emergency_exit is True
        
        # Check exit signal
        exit_reason = strategy.custom_exit(
            pair="BTC/USDT",
            trade=MagicMock(),
            current_time=datetime.now(),
            current_rate=50000.0,
            current_profit=-0.05
        )
        
        assert exit_reason == "emergency_exit"

    def test_liquidation_guard_rejection(self, strategy):
        """Test that Liquidation Guard rejects high leverage dangerous trades."""
        # Setup risky trade: Long with SL below liquidation price
        # Entry: 50000
        # Lev: 10x
        # Liq approx: 45000 (roughly)
        # SL: 40000 (Way below liq)
        
        # We need to mock the strategy stoploss to simulate this
        strategy.stoploss = -0.20 # 20% drop -> 40000
        
        # Inject config with high leverage
        strategy.config['leverage'] = 10.0
        
        # Attempt Trade
        allowed = strategy.confirm_trade_entry(
            pair="ETH/USDT",
            order_type="market",
            amount=1.0,
            rate=50000.0,
            time_in_force="gtc",
            current_time=datetime.now(),
            entry_tag="risky_long",
            side="long"
        )
        
        # Should be rejected because SL (40k) < Liq (~45k)
        assert allowed is False, "Liquidation guard should reject dangerous trade"
