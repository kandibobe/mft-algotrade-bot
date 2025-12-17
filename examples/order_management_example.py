"""
Order Management System - Usage Example
=======================================

Demonstrates how to use the order management system.

Author: Stoic Citadel Team
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.order_manager import (
    MarketOrder,
    LimitOrder,
    StopLossOrder,
    OrderSide,
    PositionManager,
    PositionSide,
    CircuitBreaker,
    CircuitBreakerConfig,
    OrderExecutor,
    ExecutionMode,
    SlippageSimulator,
    SlippageModel,
)


def example_basic_order_flow():
    """Example: Basic order creation and lifecycle."""
    print("=" * 70)
    print("Example 1: Basic Order Flow")
    print("=" * 70)

    # Create a market order
    order = MarketOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.1,
        strategy_name="StoicEnsembleV2",
    )

    print(f"\n1. Order Created:")
    print(f"   ID: {order.order_id}")
    print(f"   Symbol: {order.symbol}")
    print(f"   Side: {order.side.value}")
    print(f"   Quantity: {order.quantity}")
    print(f"   Status: {order.status.value}")

    # Simulate order submission
    from src.order_manager.order_types import OrderStatus
    order.update_status(OrderStatus.SUBMITTED)
    print(f"\n2. Order Submitted")
    print(f"   Status: {order.status.value}")

    # Simulate partial fill
    order.update_status(OrderStatus.OPEN)
    order.update_fill(filled_qty=0.05, fill_price=50000.0, commission=2.5)
    print(f"\n3. Partial Fill:")
    print(f"   Filled: {order.filled_quantity} / {order.quantity}")
    print(f"   Fill %: {order.fill_percentage:.1f}%")
    print(f"   Avg Price: ${order.average_fill_price:,.2f}")

    # Simulate complete fill
    order.update_fill(filled_qty=0.05, fill_price=50100.0, commission=2.5)
    print(f"\n4. Order Filled:")
    print(f"   Filled: {order.filled_quantity} / {order.quantity}")
    print(f"   Status: {order.status.value}")
    print(f"   Avg Price: ${order.average_fill_price:,.2f}")
    print(f"   Total Commission: ${order.commission:.2f}")


def example_position_management():
    """Example: Position tracking and PnL calculation."""
    print("\n\n" + "=" * 70)
    print("Example 2: Position Management")
    print("=" * 70)

    # Initialize position manager
    position_manager = PositionManager(
        max_positions=5,
        max_position_per_symbol=2
    )

    print(f"\nPosition Manager initialized:")
    print(f"   Max positions: {position_manager.max_positions}")
    print(f"   Max per symbol: {position_manager.max_position_per_symbol}")

    # Open a position
    position = position_manager.open_position(
        symbol="BTC/USDT",
        side=PositionSide.LONG,
        entry_price=50000.0,
        quantity=0.1,
        stop_loss=48000.0,
        take_profit=52000.0,
        strategy_name="StoicEnsembleV2",
        entry_reason="RSI oversold + MACD bullish cross",
    )

    print(f"\n1. Position Opened:")
    print(f"   ID: {position.position_id}")
    print(f"   Symbol: {position.symbol}")
    print(f"   Side: {position.side.value}")
    print(f"   Entry: ${position.entry_price:,.2f}")
    print(f"   Quantity: {position.quantity}")
    print(f"   Stop Loss: ${position.stop_loss:,.2f}")
    print(f"   Take Profit: ${position.take_profit:,.2f}")

    # Update price (profit)
    position.update_price(51000.0)
    print(f"\n2. Price Updated to $51,000:")
    print(f"   Unrealized PnL: ${position.unrealized_pnl:,.2f}")
    print(f"   Unrealized PnL %: {position.unrealized_pnl_pct:.2f}%")
    print(f"   Is Profitable: {position.is_profitable}")

    # Check if should take profit
    position.update_price(52500.0)
    print(f"\n3. Price Updated to $52,500:")
    print(f"   Should Take Profit: {position.should_take_profit}")
    print(f"   Unrealized PnL: ${position.unrealized_pnl:,.2f}")

    # Close position
    position_manager.close_position(
        position_id=position.position_id,
        exit_price=52500.0,
        exit_commission=5.25,
        reason="Take profit triggered"
    )

    print(f"\n4. Position Closed:")
    print(f"   Exit Price: ${position.exit_price:,.2f}")
    print(f"   Realized PnL: ${position.realized_pnl:,.2f}")
    print(f"   Duration: {position.duration_minutes:.1f} minutes")
    print(f"   Exit Reason: {position.exit_reason}")

    # Statistics
    stats = position_manager.get_statistics()
    print(f"\n5. Position Manager Statistics:")
    print(f"   Open Positions: {stats['open_positions']}")
    print(f"   Closed Positions: {stats['closed_positions']}")
    print(f"   Total Realized PnL: ${stats['total_realized_pnl']:,.2f}")


def example_circuit_breaker():
    """Example: Circuit breaker protecting against losses."""
    print("\n\n" + "=" * 70)
    print("Example 3: Circuit Breaker")
    print("=" * 70)

    # Configure circuit breaker
    config = CircuitBreakerConfig(
        max_daily_loss_pct=5.0,
        max_drawdown_pct=15.0,
        max_consecutive_losses=5,
        max_orders_per_minute=10,
    )

    breaker = CircuitBreaker(config)

    print(f"\nCircuit Breaker Configuration:")
    print(f"   Max Daily Loss: {config.max_daily_loss_pct}%")
    print(f"   Max Drawdown: {config.max_drawdown_pct}%")
    print(f"   Max Consecutive Losses: {config.max_consecutive_losses}")

    # Simulate trading activity
    print(f"\n1. Recording Trades:")

    # Winning trade
    breaker.record_trade(pnl=100.0)
    print(f"   Trade +$100 | Consecutive Losses: {breaker.consecutive_losses}")

    # Losing trades
    breaker.record_trade(pnl=-50.0)
    print(f"   Trade -$50  | Consecutive Losses: {breaker.consecutive_losses}")
    breaker.record_trade(pnl=-75.0)
    print(f"   Trade -$75  | Consecutive Losses: {breaker.consecutive_losses}")

    status = breaker.get_status()
    print(f"\n2. Status Check:")
    print(f"   State: {status['state']}")
    print(f"   Is Operational: {status['is_operational']}")
    print(f"   Daily PnL: ${status['daily_pnl']:.2f}")
    print(f"   Consecutive Losses: {status['consecutive_losses']}")

    # Simulate excessive loss (should trip)
    print(f"\n3. Simulating Excessive Loss:")
    breaker.update_balance(current=9000, peak=10000)

    tripped = breaker.check_and_trip(
        current_pnl=-6.0,  # Exceeds 5% limit
        current_drawdown=-10.0
    )

    print(f"   Circuit Breaker Tripped: {tripped}")
    print(f"   State: {breaker.state.value}")
    print(f"   Reason: {breaker.last_trip.reason.value if breaker.last_trip else 'N/A'}")

    # Try to check if can trade
    print(f"\n4. Can Trade?")
    print(f"   Is Operational: {breaker.is_operational}")
    print(f"   Is Tripped: {breaker.is_tripped}")

    # Manual reset
    print(f"\n5. Manual Reset:")
    breaker.reset(manual=True)
    print(f"   State: {breaker.state.value}")
    print(f"   Is Operational: {breaker.is_operational}")


def example_slippage_simulation():
    """Example: Slippage simulation for backtesting."""
    print("\n\n" + "=" * 70)
    print("Example 4: Slippage Simulation")
    print("=" * 70)

    # Create slippage simulator
    simulator = SlippageSimulator(model=SlippageModel.REALISTIC)

    print(f"\nSlippage Simulator: {SlippageModel.REALISTIC.value}")

    # Create a market order
    order = MarketOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.5,
    )

    # Market conditions
    market_price = 50000.0
    volume_24h = 1000000000.0  # $1B daily volume
    spread_pct = 0.02  # 0.02% spread

    print(f"\nMarket Conditions:")
    print(f"   Market Price: ${market_price:,.2f}")
    print(f"   24h Volume: ${volume_24h:,.0f}")
    print(f"   Spread: {spread_pct}%")

    # Simulate execution
    execution_price, commission = simulator.simulate_execution(
        order=order,
        market_price=market_price,
        volume_24h=volume_24h,
        spread_pct=spread_pct,
    )

    slippage = abs(execution_price - market_price)
    slippage_pct = (slippage / market_price) * 100

    print(f"\nExecution Results:")
    print(f"   Market Price: ${market_price:,.2f}")
    print(f"   Execution Price: ${execution_price:,.2f}")
    print(f"   Slippage: ${slippage:.2f} ({slippage_pct:.3f}%)")
    print(f"   Commission: ${commission:.2f}")

    # Validate order size
    order_value = order.quantity * market_price
    is_valid, warning = simulator.validate_order_size(
        order_value=order_value,
        volume_24h=volume_24h,
        max_slippage_pct=0.5
    )

    print(f"\nOrder Validation:")
    print(f"   Order Value: ${order_value:,.2f}")
    print(f"   Is Valid: {is_valid}")
    if warning:
        print(f"   Warning: {warning}")


def example_complete_workflow():
    """Example: Complete trading workflow."""
    print("\n\n" + "=" * 70)
    print("Example 5: Complete Trading Workflow")
    print("=" * 70)

    # Initialize components
    circuit_breaker = CircuitBreaker()
    position_manager = PositionManager(max_positions=3)
    slippage_simulator = SlippageSimulator()

    executor = OrderExecutor(
        mode=ExecutionMode.BACKTEST,
        circuit_breaker=circuit_breaker,
        slippage_simulator=slippage_simulator,
    )

    print(f"\n1. System Initialized")
    print(f"   Execution Mode: {executor.mode.value}")
    print(f"   Circuit Breaker: Active")
    print(f"   Position Manager: Max {position_manager.max_positions} positions")

    # Check if can trade
    if not circuit_breaker.is_operational:
        print("\n❌ Trading halted - circuit breaker is open!")
        return

    print(f"\n2. Creating Buy Order...")
    order = MarketOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.1,
    )

    # Execute order
    market_data = {
        "price": 50000.0,
        "close": 50000.0,
        "volume_24h": 1000000000.0,
        "spread_pct": 0.02,
    }

    result = executor.execute(order, market_data=market_data)

    print(f"\n3. Order Execution Result:")
    print(f"   Success: {result.success}")
    if result.success:
        print(f"   Execution Price: ${result.execution_price:,.2f}")
        print(f"   Filled Quantity: {result.filled_quantity}")
        print(f"   Commission: ${result.commission:.2f}")
        print(f"   Latency: {result.latency_ms:.1f}ms")

        # Open position
        position = position_manager.open_position(
            symbol=order.symbol,
            side=PositionSide.LONG,
            entry_price=result.execution_price,
            quantity=result.filled_quantity,
            stop_loss=result.execution_price * 0.95,  # 5% stop loss
            take_profit=result.execution_price * 1.10,  # 10% take profit
            entry_commission=result.commission,
        )

        print(f"\n4. Position Opened:")
        print(f"   Position ID: {position.position_id}")
        print(f"   Entry: ${position.entry_price:,.2f}")
        print(f"   Stop Loss: ${position.stop_loss:,.2f}")
        print(f"   Take Profit: ${position.take_profit:,.2f}")

    # Executor statistics
    stats = executor.get_statistics()
    print(f"\n5. Executor Statistics:")
    print(f"   Total Executions: {stats['total_executions']}")
    print(f"   Success Rate: {stats['success_rate']:.1f}%")


if __name__ == "__main__":
    print("\nStoic Citadel - Order Management System Examples\n")

    example_basic_order_flow()
    example_position_management()
    example_circuit_breaker()
    example_slippage_simulation()
    example_complete_workflow()

    print("\n\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70 + "\n")
