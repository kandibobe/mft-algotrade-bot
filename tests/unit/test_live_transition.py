
import pytest
import os
import sys
from pathlib import Path

# Add project root to sys.path to ensure absolute imports work
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.risk.risk_manager import RiskManager
from src.risk.circuit_breaker import CircuitBreaker, TripReason, CircuitState
from src.config.unified_config import load_config
from src.risk.position_sizing import PositionSizingConfig

def test_regime_scaling():
    print("\nRunning test_regime_scaling...")
    sizing_config = PositionSizingConfig(max_position_pct=1.0)
    rm = RiskManager(sizing_config=sizing_config)
    rm.initialize(100000)
    
    # Test Grind Regime (1.5x)
    res_grind = rm.evaluate_trade(
        symbol="BTC/USDT",
        entry_price=50000.0,
        stop_loss_price=49000.0,
        regime_score=0.7 # Grind
    )
    
    # Test Pump/Dump Regime (0.8x)
    res_pump = rm.evaluate_trade(
        symbol="BTC/USDT",
        entry_price=50000.0,
        stop_loss_price=49000.0,
        regime_score=0.9 # Pump/Dump
    )
    
    print(f"Grind Value: {res_grind['position_value']}, Pump Value: {res_pump['position_value']}")
    assert res_grind["position_value"] > res_pump["position_value"]

def test_kill_switch_drawdown():
    print("\nRunning test_kill_switch_drawdown...")
    rm = RiskManager()
    cb = rm.circuit_breaker
    
    # Initialize session with 10000
    cb.initialize_session(10000.0, 10000.0)
    
    # Simulate 4% drawdown (threshold is 3%)
    cb.update_metrics(balance=10000.0, equity=9500.0)
    
    status = cb.get_status()
    print(f"Status after drawdown: {status['state']}")
    assert status["state"] == "DAILY_LOSS_LIMIT_OPEN"
    assert not status["can_trade"]

def test_kill_switch_consecutive_losses():
    print("\nRunning test_kill_switch_consecutive_losses...")
    rm = RiskManager()
    cb = rm.circuit_breaker
    
    cb.initialize_session(10000.0, 10000.0)
    
    # Record 5 losses
    for i in range(5):
        cb.record_trade({"symbol": "BTC/USDT", "pnl": -100.0}, -0.01)
        
    status = cb.get_status()
    print(f"Status after losses: {status['state']}, Losses: {status['consecutive_losses']}")
    assert status["state"] == "CONSECUTIVE_LOSSES_OPEN"
    assert not status["can_trade"]

if __name__ == "__main__":
    try:
        test_regime_scaling()
        test_kill_switch_drawdown()
        test_kill_switch_consecutive_losses()
        print("\n All transition tests passed!")
    except Exception as e:
        print(f"\nL Tests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
