"""
Integration Tests for Risk Management
=====================================

Verifies that RiskManager correctly loads from Unified Config
and enforces risk limits.
"""

import os

import pytest

from src.config.manager import ConfigurationManager
from src.risk.risk_manager import RiskManager


@pytest.fixture
def risk_manager():
    # Setup mock config
    os.environ["STOIC_CONFIG_PATH"] = "config/config.json.template" # fallback or use mock
    # Initialize Config Manager (mocking the singleton)
    ConfigurationManager._instance = None # Reset
    # Create RiskManager with auto-load
    return RiskManager()

def test_risk_manager_initialization(risk_manager):
    """Test that RiskManager loads defaults if config missing."""
    assert risk_manager.circuit_breaker is not None
    assert risk_manager.position_sizer is not None

def test_circuit_breaker_integration(risk_manager):
    """Test circuit breaker stops trading on huge loss."""
    risk_manager.initialize(account_balance=10000.0)

    # Record a catastrophic loss
    # Balance 10000, Loss 2000 (20%) -> Should trip Max Drawdown (default 15% or 20%)

    # 1. Enter trade
    risk_manager.record_entry("BTC/USDT", 50000, 0.2, 45000)

    # 2. Exit with loss
    risk_manager.record_exit("BTC/USDT", 40000) # 20% drop

    status = risk_manager.get_status()
    assert status["circuit_breaker"]["can_trade"] is False
    assert status["circuit_breaker"]["state"] == "open"

def test_position_sizing_check(risk_manager):
    """Test position sizing limits."""
    risk_manager.initialize(account_balance=10000.0)

    # Try to open a position larger than max_position_pct (default 10%)
    # 10000 * 0.10 = 1000 max size

    # Evaluation for 0.5 BTC @ 50000 = 25000 USD (Too big)
    result = risk_manager.evaluate_trade(
        symbol="BTC/USDT",
        entry_price=50000,
        stop_loss_price=49000, # Small stop, so risk is low, but position size is high
        leverage=1.0
    )

    # It should be capped
    allowed_size = float(result["position_size"]) * 50000
    assert allowed_size <= 1500.0 # Allow some buffer, but definitely not 25000
