"""
Tests for circuit breaker.
"""

from datetime import datetime, timedelta

import pytest

from src.order_manager.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    TripReason,
)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initializes correctly."""
        breaker = CircuitBreaker()

        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.is_tripped is False
        assert breaker.is_operational is True
        assert breaker.daily_pnl == 0.0
        assert breaker.consecutive_losses == 0

    def test_trip_on_daily_loss(self):
        """Test trip on excessive daily loss."""
        config = CircuitBreakerConfig(max_daily_loss_pct=5.0)
        breaker = CircuitBreaker(config)

        # Should trip when loss exceeds limit
        tripped = breaker.check_and_trip(
            current_pnl=-6.0,  # -6% loss
            current_drawdown=-3.0
        )

        assert tripped is True
        assert breaker.is_tripped is True
        assert breaker.last_trip.reason == TripReason.DAILY_LOSS

    def test_trip_on_max_drawdown(self):
        """Test trip on maximum drawdown."""
        config = CircuitBreakerConfig(max_drawdown_pct=15.0)
        breaker = CircuitBreaker(config)

        # Should trip when drawdown exceeds limit
        tripped = breaker.check_and_trip(
            current_pnl=-2.0,
            current_drawdown=-16.0  # -16% drawdown
        )

        assert tripped is True
        assert breaker.is_tripped is True
        assert breaker.last_trip.reason == TripReason.MAX_DRAWDOWN

    def test_trip_on_consecutive_losses(self):
        """Test trip on consecutive losing trades."""
        config = CircuitBreakerConfig(max_consecutive_losses=5)
        breaker = CircuitBreaker(config)

        # Record consecutive losses
        for _ in range(5):
            breaker.record_trade(pnl=-100.0)

        # Should trip
        tripped = breaker.check_and_trip(
            current_pnl=-2.0,
            current_drawdown=-5.0
        )

        assert tripped is True
        assert breaker.last_trip.reason == TripReason.CONSECUTIVE_LOSSES
        assert breaker.consecutive_losses == 5

    def test_consecutive_losses_reset_on_win(self):
        """Test that consecutive losses reset on winning trade."""
        breaker = CircuitBreaker()

        # Record losses
        breaker.record_trade(-100.0)
        breaker.record_trade(-100.0)
        breaker.record_trade(-100.0)
        assert breaker.consecutive_losses == 3

        # Win resets counter
        breaker.record_trade(150.0)
        assert breaker.consecutive_losses == 0

    def test_order_rate_limiting(self):
        """Test order rate limiting."""
        config = CircuitBreakerConfig(
            max_orders_per_minute=5,
            max_orders_per_hour=50
        )
        breaker = CircuitBreaker(config)

        # Record orders
        for _ in range(6):
            breaker.record_order()

        # Should trip on excessive order rate
        tripped = breaker.check_and_trip(
            current_pnl=0.0,
            current_drawdown=0.0
        )

        assert tripped is True
        assert breaker.last_trip.reason == TripReason.ORDER_RATE

    def test_manual_reset(self):
        """Test manual reset of circuit breaker."""
        breaker = CircuitBreaker()

        # Trip breaker
        breaker.check_and_trip(current_pnl=-10.0, current_drawdown=-5.0)
        assert breaker.is_tripped is True

        # Manual reset
        breaker.reset(manual=True)
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.is_tripped is False
        assert breaker.last_trip.reset_at is not None

    def test_auto_reset_after_timeout(self):
        """Test automatic reset after timeout."""
        config = CircuitBreakerConfig(
            auto_reset_after_minutes=1,
            require_manual_reset=False
        )
        breaker = CircuitBreaker(config)

        # Trip breaker
        breaker.check_and_trip(current_pnl=-10.0, current_drawdown=-5.0)
        assert breaker.is_tripped is True

        # Simulate time passing
        breaker.last_trip.timestamp = datetime.now() - timedelta(minutes=2)

        # Check operational status (should auto-reset)
        is_operational = breaker.is_operational
        assert is_operational is True
        assert breaker.state == CircuitBreakerState.CLOSED

    def test_manual_reset_required(self):
        """Test that manual reset can be required."""
        config = CircuitBreakerConfig(
            auto_reset_after_minutes=1,
            require_manual_reset=True
        )
        breaker = CircuitBreaker(config)

        # Trip breaker
        breaker.check_and_trip(current_pnl=-10.0, current_drawdown=-5.0)

        # Simulate time passing
        breaker.last_trip.timestamp = datetime.now() - timedelta(minutes=2)

        # Should NOT auto-reset
        is_operational = breaker.is_operational
        assert is_operational is False
        assert breaker.is_tripped is True

        # Manual reset required
        breaker.reset(manual=True)
        assert breaker.is_operational is True

    def test_error_tracking(self):
        """Test system error tracking."""
        config = CircuitBreakerConfig(max_consecutive_errors=3)
        breaker = CircuitBreaker(config)

        # Record errors
        breaker.record_error("Connection timeout")
        breaker.record_error("API error")
        assert breaker.consecutive_errors == 2

        # Clear errors on success
        breaker.clear_errors()
        assert breaker.consecutive_errors == 0

        # Record errors again
        breaker.record_error("Error 1")
        breaker.record_error("Error 2")
        breaker.record_error("Error 3")

        # Should trip after max errors
        assert breaker.is_tripped is True
        assert breaker.last_trip.reason == TripReason.SYSTEM_ERROR

    def test_drawdown_calculation(self):
        """Test drawdown calculation."""
        breaker = CircuitBreaker()

        # Set balance
        breaker.update_balance(current=10000, peak=10000)
        assert breaker.calculate_drawdown() == 0.0

        # Simulate loss
        breaker.update_balance(current=9000)
        drawdown = breaker.calculate_drawdown()
        assert abs(drawdown - (-10.0)) < 0.01  # -10%

        # Update peak
        breaker.update_balance(current=11000)
        assert breaker.peak_balance == 11000
        assert breaker.calculate_drawdown() == 0.0

    def test_status_report(self):
        """Test status reporting."""
        breaker = CircuitBreaker()

        breaker.record_trade(-100.0)
        breaker.record_trade(-50.0)
        breaker.record_order()
        breaker.update_balance(current=9500, peak=10000)

        status = breaker.get_status()

        assert status["state"] == "closed"
        assert status["is_tripped"] is False
        assert status["is_operational"] is True
        assert status["daily_pnl"] == -150.0
        assert status["consecutive_losses"] == 2
        assert status["current_drawdown"] == -5.0
        assert status["orders_last_minute"] == 1

    def test_force_trip(self):
        """Test manual force trip."""
        breaker = CircuitBreaker()

        # Force trip
        tripped = breaker.check_and_trip(
            current_pnl=0.0,
            current_drawdown=0.0,
            force=True
        )

        assert tripped is True
        assert breaker.is_tripped is True
        assert breaker.last_trip.reason == TripReason.MANUAL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
