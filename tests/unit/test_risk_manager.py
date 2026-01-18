"""
Unit Tests for Risk Manager
===========================

Verifies that RiskManager correctly loads from Unified Config
and enforces risk limits.
"""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from src.risk.risk_manager import RiskManager


@pytest.fixture
def mock_config():
    """Mock the configuration to avoid file system dependency."""
    with patch("src.config.manager.ConfigurationManager") as MockConfig:
        cfg = MagicMock()
        cfg.risk.max_drawdown_pct = 0.15
        cfg.risk.max_daily_loss_pct = 0.05
        cfg.risk.max_position_pct = 0.10
        cfg.risk.max_portfolio_risk = 0.02
        cfg.risk.safety_buffer = 0.01

        # Configure the mock to return our config object
        MockConfig.config.return_value = cfg
        # Also handle the call() pattern if used
        MockConfig.return_value = cfg
        yield cfg

@pytest.fixture
def risk_manager(mock_config):
    """Create RiskManager with mocked config and disabled persistence."""
    # Patch CircuitBreaker persistence to avoid side effects
    with patch("src.risk.circuit_breaker.CircuitBreaker.load_state") as mock_load, \
         patch("src.risk.circuit_breaker.CircuitBreaker.save_state") as mock_save:
        rm = RiskManager(circuit_config=None, sizing_config=None, liquidation_config=None)
        yield rm

def test_risk_manager_initialization(risk_manager):
    """Test that RiskManager initializes components correctly."""
    assert risk_manager.circuit_breaker is not None
    assert risk_manager.position_sizer is not None
    assert risk_manager.liquidation_guard is not None

    # Check default state
    assert risk_manager._account_balance == Decimal("0.0")
    assert risk_manager._metrics.can_trade is True

def test_circuit_breaker_integration(risk_manager):
    """Test circuit breaker stops trading on huge loss."""
    risk_manager.initialize(account_balance=10000.0)

    # 1. Enter trade
    risk_manager.record_entry("BTC/USDT", 50000, 0.2, 45000)

    # 2. Exit with catastrophic loss (20% drop)
    # Balance 10000, Position 0.2 BTC * 50000 = 10000 (Full port, hypothetical)
    # Exit at 40000 -> Loss = (40000 - 50000) * 0.2 = -2000
    # Drawdown = 2000 / 10000 = 20% > 15% limit
    risk_manager.record_exit("BTC/USDT", 40000)

    status = risk_manager.get_status()

    # Circuit breaker should be open (trading halted)
    assert status["circuit_breaker"]["can_trade"] is False
    assert status["circuit_breaker"]["state"] == "open"

    # Verify evaluate_trade rejects new trades
    res = risk_manager.evaluate_trade("ETH/USDT", 3000, 2900)
    assert res["allowed"] is False
    assert "Circuit breaker is OPEN" in res["rejection_reason"]

def test_position_sizing_check(risk_manager):
    """Test position sizing limits."""
    risk_manager.initialize(account_balance=10000.0)

    # Try to open a position larger than max_position_pct (10% = 1000 USDT)
    # Asking for 0.5 BTC @ 50000 = 25000 USDT (Way too big)
    result = risk_manager.evaluate_trade(
        symbol="BTC/USDT",
        entry_price=50000,
        stop_loss_price=49000,
    )

    # It should be capped at max_position_pct * balance
    # 10000 * 0.10 = 1000 USDT
    expected_max_value = 1000.0

    assert result["allowed"] is True
    # Allow small floating point margin or rounding differences
    assert float(result["position_value"]) <= expected_max_value * 1.01

    # Ensure it's not the requested huge size
    assert float(result["position_value"]) < 20000.0

def test_emergency_stop(risk_manager):
    """Test manual emergency stop."""
    risk_manager.initialize(account_balance=10000.0)

    assert risk_manager.evaluate_trade("BTC/USDT", 50000, 49000)["allowed"] is True

    risk_manager.emergency_stop()

    res = risk_manager.evaluate_trade("BTC/USDT", 50000, 49000)
    assert res["allowed"] is False
    assert "Emergency Stop Active" in res["rejection_reason"]
