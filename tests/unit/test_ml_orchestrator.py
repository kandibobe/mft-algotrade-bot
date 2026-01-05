import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import time
from src.ml.training.orchestrator import AutomatedRetrainingOrchestrator, OrchestratorConfig

@pytest.fixture
def orchestrator():
    config = OrchestratorConfig(
        performance_window_trades=5, # Small window for testing
        min_accuracy=0.6,
        max_drawdown_window=0.05,
        enable_auto_retraining=True,
        cooldown_minutes=0 # Disable cooldown for testing trigger
    )
    return AutomatedRetrainingOrchestrator(config)

def test_metrics_calculation(orchestrator):
    """Test that rolling metrics are calculated correctly."""
    # Add 3 winning trades
    orchestrator.record_trade_result({'pnl': 10, 'pnl_pct': 0.01})
    orchestrator.record_trade_result({'pnl': 20, 'pnl_pct': 0.02})
    orchestrator.record_trade_result({'pnl': 15, 'pnl_pct': 0.015})
    
    assert orchestrator.rolling_accuracy == 1.0
    assert orchestrator.rolling_profit == 0.045
    
    # Add 2 losing trades
    orchestrator.record_trade_result({'pnl': -5, 'pnl_pct': -0.005})
    orchestrator.record_trade_result({'pnl': -10, 'pnl_pct': -0.01})
    
    # Total 5 trades: 3 wins, 2 losses. Accuracy = 3/5 = 0.6
    assert orchestrator.rolling_accuracy == 0.6
    # Profit = 0.045 - 0.015 = 0.03
    assert abs(orchestrator.rolling_profit - 0.03) < 1e-6

def test_window_sliding(orchestrator):
    """Test that metrics are calculated over the sliding window."""
    # Window size is 5
    
    # Fill window with wins
    for _ in range(5):
        orchestrator.record_trade_result({'pnl': 10, 'pnl_pct': 0.01})
        
    assert orchestrator.rolling_accuracy == 1.0
    
    # Add a loss, should push out the first win
    orchestrator.record_trade_result({'pnl': -10, 'pnl_pct': -0.01})
    
    # Now window has 4 wins, 1 loss. Accuracy = 0.8
    assert len(orchestrator.trade_history) == 5
    assert orchestrator.rolling_accuracy == 0.8

def test_accuracy_drift_detection(orchestrator):
    """Test that low accuracy triggers retraining."""
    # Mock the retraining method
    orchestrator._execute_retraining_mock = MagicMock()
    
    # Config min_accuracy is 0.6
    
    # Add 5 trades resulting in 0.4 accuracy (2 wins, 3 losses)
    trades = [
        {'pnl': 10}, {'pnl': 10}, # Wins
        {'pnl': -10}, {'pnl': -10}, {'pnl': -10} # Losses
    ]
    
    for t in trades:
        orchestrator.record_trade_result(t)
        
    # After retraining, metrics are reset
    assert orchestrator.rolling_accuracy == 0.0
    assert orchestrator._execute_retraining_mock.called

def test_profit_drift_detection(orchestrator):
    """Test that high drawdown triggers retraining."""
    orchestrator._execute_retraining_mock = MagicMock()
    
    # Config max_drawdown_window is 0.05 (5%)
    
    # Add trades leading to -6% profit
    trades = [
        {'pnl': -10, 'pnl_pct': -0.02},
        {'pnl': -10, 'pnl_pct': -0.02},
        {'pnl': -10, 'pnl_pct': -0.02},
        {'pnl': 0, 'pnl_pct': 0}, # Fill window
        {'pnl': 0, 'pnl_pct': 0}
    ]
    
    for t in trades:
        orchestrator.record_trade_result(t)
        
    assert orchestrator.rolling_profit == -0.06
    assert orchestrator._execute_retraining_mock.called

def test_retraining_cooldown(orchestrator):
    """Test that retraining respects cooldown period."""
    orchestrator._execute_retraining_mock = MagicMock()
    orchestrator.config.cooldown_minutes = 60
    
    # Trigger first retraining
    orchestrator.trigger_retraining("test")
    assert orchestrator._execute_retraining_mock.call_count == 1
    
    # Try triggering again immediately
    orchestrator.trigger_retraining("test")
    assert orchestrator._execute_retraining_mock.call_count == 1 # Should not increment
    
    # Simulate time passing (hacky manual set for test)
    orchestrator.last_retrain_time = datetime.utcnow() - timedelta(minutes=61)
    
    # Try triggering again
    orchestrator.trigger_retraining("test")
    assert orchestrator._execute_retraining_mock.call_count == 2
