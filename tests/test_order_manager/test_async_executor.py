"""
Tests for Async Smart Limit Executor
====================================

Comprehensive test suite covering:
1. Normal order flow (place -> fill)
2. Order chasing logic
3. Timeout and fallback to market
4. Race conditions (order fills during cancel)
5. Retry logic with network failures (using tenacity)
6. Partial fills
7. Edge cases
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

from src.order_manager.smart_limit_executor import (
    AsyncSmartLimitExecutor,
    SmartLimitConfig,
    ExecutionState,
    SmartExecutionResult,
)
from src.order_manager.order_types import Order, OrderType, OrderSide


@pytest.fixture
def mock_exchange():
    """Create a mock exchange API."""
    exchange = AsyncMock()

    # Default successful order responses
    exchange.create_limit_order = AsyncMock(return_value={
        "id": "order_123",
        "status": "open",
        "filled": 0,
        "average": 0,
    })

    exchange.fetch_order = AsyncMock(return_value={
        "id": "order_123",
        "status": "closed",
        "filled": 1.0,
        "average": 50000.0,
        "fee": {"cost": 10.0},
    })

    exchange.cancel_order = AsyncMock(return_value={"status": "canceled"})

    exchange.create_market_order = AsyncMock(return_value={
        "id": "market_order_456",
        "status": "closed",
        "filled": 1.0,
        "average": 50000.0,
    })

    exchange.fetch_order_book = Mock(return_value={
        "bids": [[49900, 10], [49800, 20]],
        "asks": [[50100, 10], [50200, 20]],
    })

    return exchange


@pytest.fixture
def sample_order():
    """Create a sample buy order."""
    return Order(
        order_id="test_order_1",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1.0,
        price=50000.0,
    )


@pytest.fixture
def fast_config():
    """Config for fast testing (short timeouts)."""
    return SmartLimitConfig(
        max_wait_seconds=2.0,
        chase_interval_seconds=0.5,
        convert_to_market=True,
        market_conversion_threshold=0.8,
    )


class TestBasicOrderFlow:
    """Test normal order execution flow."""

    @pytest.mark.asyncio
    async def test_successful_maker_fill(self, mock_exchange, sample_order, fast_config):
        """Test successful order fill as maker."""
        executor = AsyncSmartLimitExecutor(fast_config)

        orderbook = {
            "bids": [[49900, 10]],
            "asks": [[50100, 10]],
        }

        result = await executor.execute_async(
            order=sample_order,
            exchange_api=mock_exchange,
            orderbook=orderbook
        )

        assert result.success is True
        assert result.execution_type == "maker"
        assert result.filled_quantity > 0
        assert result.fee_saved_vs_market > 0  # Should save fees vs market order
        assert executor.state == ExecutionState.FILLED

    @pytest.mark.asyncio
    async def test_order_placement(self, mock_exchange, sample_order, fast_config):
        """Test that order is placed with correct parameters."""
        executor = AsyncSmartLimitExecutor(fast_config)

        orderbook = {
            "bids": [[49900, 10]],
            "asks": [[50100, 10]],
        }

        await executor.execute_async(
            order=sample_order,
            exchange_api=mock_exchange,
            orderbook=orderbook
        )

        # Verify create_limit_order was called
        mock_exchange.create_limit_order.assert_called_once()

        # Check call arguments
        call_args = mock_exchange.create_limit_order.call_args
        assert call_args.kwargs['symbol'] == "BTC/USDT"
        assert call_args.kwargs['side'] == "buy"
        assert call_args.kwargs['amount'] == 1.0


class TestOrderChasing:
    """Test order chasing logic."""

    @pytest.mark.asyncio
    async def test_order_chase_on_timeout(self, mock_exchange, sample_order, fast_config):
        """Test that order price is updated (chased) when not filled."""
        # Make order stay open for a while
        fetch_call_count = 0

        async def fetch_order_side_effect(order_id, symbol):
            nonlocal fetch_call_count
            fetch_call_count += 1

            if fetch_call_count < 2:
                # First check: still open
                return {"status": "open", "filled": 0, "average": 0}
            else:
                # Eventually fill
                return {"status": "closed", "filled": 1.0, "average": 50000.0}

        mock_exchange.fetch_order.side_effect = fetch_order_side_effect

        executor = AsyncSmartLimitExecutor(fast_config)

        orderbook = {
            "bids": [[49900, 10]],
            "asks": [[50100, 10]],
        }

        result = await executor.execute_async(
            order=sample_order,
            exchange_api=mock_exchange,
            orderbook=orderbook
        )

        # Should eventually fill
        assert result.success is True

        # Should have chased at least once
        # (cancel called when chasing)
        assert mock_exchange.cancel_order.call_count >= 1

    @pytest.mark.asyncio
    async def test_chase_price_calculation(self, mock_exchange, sample_order, fast_config):
        """Test that chase price moves toward market."""
        executor = AsyncSmartLimitExecutor(fast_config)

        # Initial orderbook
        orderbook = {
            "bids": [[49900, 10]],
            "asks": [[50100, 10]],
        }

        # Calculate initial price
        initial_price = executor._calculate_limit_price(
            sample_order, 49900, 50100
        )

        # Simulate market moving up
        new_orderbook = {
            "bids": [[50000, 10]],  # Bid moved up
            "asks": [[50200, 10]],
        }

        # Calculate chase price
        chase_price = executor._calculate_chase_price(
            sample_order, initial_price, new_orderbook, chase_count=1
        )

        # Chase price should be higher than initial (following market)
        assert chase_price > initial_price, "Chase price should increase with rising market"


class TestTimeoutAndFallback:
    """Test timeout handling and market order fallback."""

    @pytest.mark.asyncio
    async def test_timeout_converts_to_market(self, mock_exchange, sample_order):
        """Test that unfilled order converts to market after timeout."""
        # Order never fills
        mock_exchange.fetch_order.return_value = {
            "status": "open",
            "filled": 0,
            "average": 0
        }

        config = SmartLimitConfig(
            max_wait_seconds=1.0,  # Short timeout
            chase_interval_seconds=0.3,
            convert_to_market=True,
            market_conversion_threshold=0.5,  # Convert after 50% of timeout
        )

        executor = AsyncSmartLimitExecutor(config)

        orderbook = {
            "bids": [[49900, 10]],
            "asks": [[50100, 10]],
        }

        result = await executor.execute_async(
            order=sample_order,
            exchange_api=mock_exchange,
            orderbook=orderbook
        )

        # Should convert to market and succeed
        assert result.success is True
        assert result.execution_type in ("taker", "partial")

        # Market order should have been called
        mock_exchange.create_market_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_timeout_without_market_conversion(self, mock_exchange, sample_order):
        """Test timeout behavior when market conversion is disabled."""
        mock_exchange.fetch_order.return_value = {
            "status": "open",
            "filled": 0,
            "average": 0
        }

        config = SmartLimitConfig(
            max_wait_seconds=1.0,
            chase_interval_seconds=0.3,
            convert_to_market=False,  # Don't convert
        )

        executor = AsyncSmartLimitExecutor(config)

        orderbook = {
            "bids": [[49900, 10]],
            "asks": [[50100, 10]],
        }

        result = await executor.execute_async(
            order=sample_order,
            exchange_api=mock_exchange,
            orderbook=orderbook
        )

        # Should fail (no market conversion)
        assert result.success is False
        assert result.error_message is not None


class TestRaceConditions:
    """Test race condition handling (critical!)."""

    @pytest.mark.asyncio
    async def test_order_fills_during_cancel_attempt(self, mock_exchange, sample_order, fast_config):
        """
        Test race condition: Order fills between status check and cancel.

        Scenario:
        1. Check status -> "open"
        2. (Order fills on exchange)
        3. Attempt to cancel -> Should handle gracefully
        """
        check_count = 0

        async def fetch_order_race_condition(order_id, symbol):
            nonlocal check_count
            check_count += 1

            if check_count == 1:
                # First check in timeout handler: open
                return {"status": "open", "filled": 0, "average": 0}
            else:
                # Second check (inside lock): filled!
                return {"status": "closed", "filled": 1.0, "average": 50000.0}

        mock_exchange.fetch_order.side_effect = fetch_order_race_condition

        # Make cancel raise exception (order already gone)
        mock_exchange.cancel_order.side_effect = Exception("Order not found")

        executor = AsyncSmartLimitExecutor(fast_config)

        orderbook = {
            "bids": [[49900, 10]],
            "asks": [[50100, 10]],
        }

        result = await executor.execute_async(
            order=sample_order,
            exchange_api=mock_exchange,
            orderbook=orderbook
        )

        # Should handle race condition gracefully and succeed
        # (detected fill in second status check)
        assert result.success is True or executor.state == ExecutionState.FILLED

    @pytest.mark.asyncio
    async def test_concurrent_cancel_attempts(self, mock_exchange, sample_order, fast_config):
        """Test that lock prevents concurrent cancel attempts."""
        cancel_calls = []

        async def tracked_cancel(order_id, symbol):
            cancel_calls.append(asyncio.current_task().get_name())
            await asyncio.sleep(0.1)  # Simulate slow cancel
            return {"status": "canceled"}

        mock_exchange.cancel_order.side_effect = tracked_cancel
        mock_exchange.fetch_order.return_value = {
            "status": "open", "filled": 0, "average": 0
        }

        executor = AsyncSmartLimitExecutor(fast_config)

        orderbook = {
            "bids": [[49900, 10]],
            "asks": [[50100, 10]],
        }

        # This should timeout and attempt cancel
        result = await executor.execute_async(
            order=sample_order,
            exchange_api=mock_exchange,
            orderbook=orderbook
        )

        # Lock should prevent concurrent cancels - only one cancel call
        # (though multiple checks might happen)
        assert len(cancel_calls) <= 2, "Should not have many concurrent cancel attempts"


class TestPartialFills:
    """Test partial fill handling."""

    @pytest.mark.asyncio
    async def test_partial_fill_tracked(self, mock_exchange, sample_order, fast_config):
        """Test that partial fills are tracked correctly."""
        check_count = 0

        async def fetch_with_partial(order_id, symbol):
            nonlocal check_count
            check_count += 1

            if check_count == 1:
                # First check: partially filled
                return {"status": "partially_filled", "filled": 0.5, "average": 50000.0}
            else:
                # Eventually fully filled
                return {"status": "closed", "filled": 1.0, "average": 50000.0}

        mock_exchange.fetch_order.side_effect = fetch_with_partial

        executor = AsyncSmartLimitExecutor(fast_config)

        orderbook = {
            "bids": [[49900, 10]],
            "asks": [[50100, 10]],
        }

        result = await executor.execute_async(
            order=sample_order,
            exchange_api=mock_exchange,
            orderbook=orderbook
        )

        assert result.success is True
        assert result.filled_quantity == 1.0  # Eventually fully filled


class TestRetryLogic:
    """Test retry logic with network failures."""

    @pytest.mark.asyncio
    async def test_retry_on_network_error(self, mock_exchange, sample_order, fast_config):
        """Test that network errors trigger retry."""
        attempt_count = 0

        async def create_order_with_retry(symbol, side, amount, price):
            nonlocal attempt_count
            attempt_count += 1

            if attempt_count == 1:
                # First attempt fails
                raise ConnectionError("Network timeout")
            else:
                # Second attempt succeeds
                return {"id": "order_123", "status": "open"}

        mock_exchange.create_limit_order.side_effect = create_order_with_retry

        executor = AsyncSmartLimitExecutor(fast_config)

        orderbook = {
            "bids": [[49900, 10]],
            "asks": [[50100, 10]],
        }

        result = await executor.execute_async(
            order=sample_order,
            exchange_api=mock_exchange,
            orderbook=orderbook
        )

        # Should succeed after retry
        assert result.success is True
        assert attempt_count == 2, "Should have retried once"

    @pytest.mark.asyncio
    async def test_non_retryable_error_fails_immediately(self, mock_exchange, sample_order, fast_config):
        """Test that non-retryable errors don't trigger retry."""
        mock_exchange.create_limit_order.side_effect = ValueError("Invalid parameters")

        executor = AsyncSmartLimitExecutor(fast_config)

        orderbook = {
            "bids": [[49900, 10]],
            "asks": [[50100, 10]],
        }

        result = await executor.execute_async(
            order=sample_order,
            exchange_api=mock_exchange,
            orderbook=orderbook
        )

        # Should fail without retry
        assert result.success is False


class TestEdgeCases:
    """Test edge cases."""

    @pytest.mark.asyncio
    async def test_empty_orderbook(self, mock_exchange, sample_order, fast_config):
        """Test handling of empty orderbook."""
        executor = AsyncSmartLimitExecutor(fast_config)

        # Empty orderbook
        orderbook = {
            "bids": [],
            "asks": [],
        }

        result = await executor.execute_async(
            order=sample_order,
            exchange_api=mock_exchange,
            orderbook=orderbook
        )

        assert result.success is False
        assert "order book" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_wide_spread_fallback(self, mock_exchange, sample_order, fast_config):
        """Test that wide spread triggers market order fallback."""
        executor = AsyncSmartLimitExecutor(fast_config)

        # Very wide spread (10%)
        orderbook = {
            "bids": [[45000, 10]],
            "asks": [[55000, 10]],
        }

        result = await executor.execute_async(
            order=sample_order,
            exchange_api=mock_exchange,
            orderbook=orderbook
        )

        # Should fall back to market order due to wide spread
        assert result.success is True
        mock_exchange.create_market_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_state_transitions(self, mock_exchange, sample_order, fast_config):
        """Test that state machine transitions correctly."""
        executor = AsyncSmartLimitExecutor(fast_config)

        states_seen = []

        def track_state(state, data):
            states_seen.append(state)

        orderbook = {
            "bids": [[49900, 10]],
            "asks": [[50100, 10]],
        }

        await executor.execute_async(
            order=sample_order,
            exchange_api=mock_exchange,
            orderbook=orderbook,
            on_state_change=track_state
        )

        # Should have seen PLACED and FILLED states
        assert ExecutionState.PLACED in states_seen
        assert executor.state == ExecutionState.FILLED


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
