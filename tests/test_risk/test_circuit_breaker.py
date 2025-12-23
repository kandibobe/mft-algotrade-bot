"""
Tests for Circuit Breaker System
=================================
"""

import pytest
from datetime import datetime, timedelta

from src.risk.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    TripReason
)


class TestCircuitBreaker:
    """Test suite for CircuitBreaker."""
    
    @pytest.fixture
    def breaker(self):
        """Create a circuit breaker with test config."""
        config = CircuitBreakerConfig(
            daily_loss_limit_pct=0.05,
            consecutive_loss_limit=3,
            max_drawdown_pct=0.10,
            cooldown_minutes=1
        )
        cb = CircuitBreaker(config)
        cb.initialize_session(10000.0)
        return cb
    
    def test_initial_state(self, breaker):
        """Test initial circuit breaker state."""
        assert breaker.state == CircuitState.CLOSED
        assert breaker.can_trade() is True
        assert breaker.get_position_multiplier() == 1.0
    
    def test_daily_loss_limit_trip(self, breaker):
        """Test trip on daily loss limit."""
        # Simulate 5% loss
        breaker.record_trade({"symbol": "BTC/USDT"}, -0.05)
        
        assert breaker.state == CircuitState.OPEN
        assert breaker.trip_reason == TripReason.DAILY_LOSS_LIMIT
        assert breaker.can_trade() is False
    
    def test_consecutive_losses_trip(self, breaker):
        """Test trip on consecutive losses."""
        # Simulate 3 consecutive losses
        for i in range(3):
            breaker.record_trade({"symbol": "BTC/USDT"}, -0.01)
        
        assert breaker.state == CircuitState.OPEN
        assert breaker.trip_reason == TripReason.CONSECUTIVE_LOSSES
    
    def test_consecutive_losses_reset_on_win(self, breaker):
        """Test consecutive losses counter resets on win."""
        breaker.record_trade({"symbol": "BTC/USDT"}, -0.01)
        breaker.record_trade({"symbol": "BTC/USDT"}, -0.01)
        breaker.record_trade({"symbol": "BTC/USDT"}, 0.02)  # Win
        
        assert breaker.session.consecutive_losses == 0
        assert breaker.state == CircuitState.CLOSED
    
    def test_drawdown_trip(self, breaker):
        """Test trip on max drawdown."""
        # Simulate profit then big loss
        breaker.update_balance(11000.0)  # New peak
        breaker.update_balance(9800.0)   # 10.9% drawdown
        breaker.record_trade({"symbol": "BTC/USDT"}, -0.001)  # Small loss triggers check
        
        assert breaker.state == CircuitState.OPEN
        assert breaker.trip_reason == TripReason.MAX_DRAWDOWN
    
    def test_manual_stop(self, breaker):
        """Test manual emergency stop."""
        breaker.manual_stop()
        
        assert breaker.state == CircuitState.OPEN
        assert breaker.trip_reason == TripReason.MANUAL_STOP
    
    def test_manual_reset(self, breaker):
        """Test manual reset after trip."""
        breaker.manual_stop()
        assert breaker.state == CircuitState.OPEN
        
        breaker.manual_reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.can_trade() is True
    
    def test_position_multiplier_in_recovery(self, breaker):
        """Test reduced position size in half-open state."""
        breaker.manual_stop()
        
        # Simulate cooldown passing
        breaker.trip_time = datetime.utcnow() - timedelta(minutes=5)
        breaker.can_trade()  # This should transition to half-open
        
        assert breaker.state == CircuitState.HALF_OPEN
        assert breaker.get_position_multiplier() == 0.25  # Recovery size
    
    def test_recovery_success(self, breaker):
        """Test successful recovery from half-open."""
        breaker.config.recovery_trades_required = 2
        breaker.manual_stop()
        breaker.trip_time = datetime.utcnow() - timedelta(minutes=5)
        breaker.can_trade()  # Transition to half-open
        
        # Successful recovery trades
        breaker.attempt_recovery(trade_successful=True)
        breaker.attempt_recovery(trade_successful=True)
        
        assert breaker.state == CircuitState.CLOSED
    
    def test_recovery_failure(self, breaker):
        """Test failed recovery goes back to open."""
        breaker.manual_stop()
        breaker.trip_time = datetime.utcnow() - timedelta(minutes=5)
        breaker.can_trade()  # Transition to half-open
        
        assert breaker.state == CircuitState.HALF_OPEN
        
        # Failed recovery trade
        breaker.attempt_recovery(trade_successful=False)
        
        assert breaker.state == CircuitState.OPEN
        assert breaker.recovery_trades == 0
