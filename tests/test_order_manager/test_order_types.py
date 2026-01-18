"""
Tests for order types and state machine.
"""


import pytest

from src.order_manager.order_types import (
    LimitOrder,
    MarketOrder,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    StopLossOrder,
    TrailingStopOrder,
)


class TestOrder:
    """Test Order class and state machine."""

    def test_order_creation(self):
        """Test basic order creation."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
        )

        assert order.symbol == "BTC/USDT"
        assert order.side == OrderSide.BUY
        assert order.quantity == 0.1
        assert order.status == OrderStatus.PENDING
        assert order.remaining_quantity == 0.1
        assert order.filled_quantity == 0.0

    def test_order_status_transitions(self):
        """Test valid status transitions."""
        order = Order(symbol="BTC/USDT", side=OrderSide.BUY, quantity=0.1)

        # PENDING → SUBMITTED
        order.update_status(OrderStatus.SUBMITTED)
        assert order.status == OrderStatus.SUBMITTED
        assert order.submitted_at is not None

        # SUBMITTED → OPEN
        order.update_status(OrderStatus.OPEN)
        assert order.status == OrderStatus.OPEN

        # OPEN → FILLED
        order.update_status(OrderStatus.FILLED)
        assert order.status == OrderStatus.FILLED
        assert order.filled_at is not None

    def test_invalid_status_transition(self):
        """Test that invalid transitions are rejected."""
        order = Order(symbol="BTC/USDT", side=OrderSide.BUY, quantity=0.1)
        order.update_status(OrderStatus.FILLED)

        # Cannot transition from FILLED (terminal state)
        order.update_status(OrderStatus.OPEN)
        assert order.status == OrderStatus.FILLED  # Should remain FILLED

    def test_order_fill(self):
        """Test order fill updates."""
        order = Order(symbol="BTC/USDT", side=OrderSide.BUY, quantity=1.0)
        order.update_status(OrderStatus.OPEN)

        # Partial fill
        order.update_fill(filled_qty=0.5, fill_price=50000.0, commission=0.5)

        assert order.filled_quantity == 0.5
        assert order.remaining_quantity == 0.5
        assert order.average_fill_price == 50000.0
        assert order.commission == 0.5
        assert order.status == OrderStatus.PARTIALLY_FILLED

        # Complete fill
        order.update_fill(filled_qty=0.5, fill_price=50100.0, commission=0.5)

        assert order.filled_quantity == 1.0
        assert order.remaining_quantity == 0.0
        assert order.status == OrderStatus.FILLED
        assert order.fill_percentage == 100.0

    def test_average_fill_price_calculation(self):
        """Test weighted average fill price."""
        order = Order(symbol="BTC/USDT", side=OrderSide.BUY, quantity=1.0)
        order.update_status(OrderStatus.OPEN)

        # Fill 1: 0.5 @ 50000
        order.update_fill(0.5, 50000.0)

        # Fill 2: 0.5 @ 50200
        order.update_fill(0.5, 50200.0)

        # Average should be 50100
        assert abs(order.average_fill_price - 50100.0) < 0.01

    def test_market_order(self):
        """Test market order creation."""
        order = MarketOrder(
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            quantity=2.0,
        )

        assert order.order_type == OrderType.MARKET
        assert order.side == OrderSide.SELL

    def test_limit_order_requires_price(self):
        """Test that limit order requires price."""
        with pytest.raises(ValueError, match="Limit order requires price"):
            LimitOrder(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                quantity=0.1,
            )

        # Should work with price
        order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0,
        )
        assert order.order_type == OrderType.LIMIT
        assert order.price == 50000.0

    def test_stop_loss_order(self):
        """Test stop-loss order."""
        with pytest.raises(ValueError, match="Stop-loss order requires stop_price"):
            StopLossOrder(
                symbol="BTC/USDT",
                side=OrderSide.SELL,
                quantity=0.1,
            )

        # Should work with stop_price
        order = StopLossOrder(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=0.1,
            stop_price=48000.0,
        )
        assert order.order_type == OrderType.STOP_LOSS
        assert order.stop_price == 48000.0

    def test_trailing_stop_order(self):
        """Test trailing stop order."""
        order = TrailingStopOrder(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=0.1,
            trailing_distance=2.0,  # 2% trailing
            trailing_percent=True,
        )

        assert order.order_type == OrderType.TRAILING_STOP

        # Update with price movement
        order.update_trailing_stop(50000.0)  # Initial price
        assert order.highest_price == 50000.0
        assert order.stop_price == 49000.0  # 50000 * (1 - 0.02)

        # Price increases, stop should follow
        order.update_trailing_stop(51000.0)
        assert order.highest_price == 51000.0
        assert order.stop_price == 49980.0  # 51000 * (1 - 0.02)

        # Price decreases, stop should NOT move down
        order.update_trailing_stop(50500.0)
        assert order.highest_price == 51000.0  # Unchanged
        assert order.stop_price == 49980.0  # Unchanged

    def test_order_retry_logic(self):
        """Test order retry counter."""
        order = Order(symbol="BTC/USDT", side=OrderSide.BUY, quantity=0.1)

        assert order.can_retry() is True
        assert order.retry_count == 0

        order.increment_retry()
        assert order.retry_count == 1
        assert order.can_retry() is True

        # Max retries
        order.max_retries = 2
        order.increment_retry()
        assert order.retry_count == 2
        assert order.can_retry() is False

    def test_order_is_properties(self):
        """Test order boolean properties."""
        order = Order(symbol="BTC/USDT", side=OrderSide.BUY, quantity=0.1)

        assert order.is_buy is True
        assert order.is_sell is False
        assert order.is_active is True
        assert order.is_terminal is False
        assert order.is_filled is False

        # Fill order
        order.update_status(OrderStatus.FILLED)

        assert order.is_active is False
        assert order.is_terminal is True
        assert order.is_filled is True

    def test_order_to_dict(self):
        """Test order serialization."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0,
        )

        data = order.to_dict()

        assert data["symbol"] == "BTC/USDT"
        assert data["side"] == "buy"
        assert data["quantity"] == 0.1
        assert data["price"] == 50000.0
        assert data["status"] == "pending"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
