# Order Management System

Comprehensive order management system for algorithmic trading with advanced risk controls.

## Features

### 1. Order Types & State Machine

- **Market Orders** - Execute immediately at best available price
- **Limit Orders** - Execute at specified price or better
- **Stop-Loss Orders** - Trigger when price crosses stop level
- **Take-Profit Orders** - Trigger at profit target
- **Trailing Stop Orders** - Stop price trails market price

### 2. Position Management

- Track multiple open positions per symbol
- Real-time PnL calculation (realized & unrealized)
- Automatic stop-loss and take-profit monitoring
- Position size limits enforcement

### 3. Circuit Breaker

Emergency stop mechanism to protect against catastrophic losses:

- **Daily Loss Limit** - Halt trading on excessive daily losses
- **Maximum Drawdown** - Stop at predefined drawdown threshold
- **Consecutive Losses** - Trip on strategy failure patterns
- **Order Rate Limiting** - Prevent runaway order loops
- **System Error Protection** - Halt on repeated errors

### 4. Slippage Simulation

Realistic execution simulation for backtesting:

- Volume-based slippage modeling
- Market impact calculation
- Bid-ask spread costs
- Commission tiers

### 5. Order Executor

Robust order execution with retry logic:

- Pre-execution validation
- Automatic retry on transient failures
- Multiple execution modes (live, paper, backtest)
- Detailed execution logging

## Installation

Already installed as part of the project. Import from:

```python
from src.order_manager import (
    MarketOrder,
    LimitOrder,
    PositionManager,
    CircuitBreaker,
    OrderExecutor,
)
```

## Quick Start

### Basic Order Flow

```python
from src.order_manager import MarketOrder, OrderSide

# Create market order
order = MarketOrder(
    symbol="BTC/USDT",
    side=OrderSide.BUY,
    quantity=0.1,
)

# Order lifecycle
order.update_status(OrderStatus.SUBMITTED)
order.update_status(OrderStatus.OPEN)
order.update_fill(filled_qty=0.1, fill_price=50000.0, commission=5.0)

print(f"Order filled @ ${order.average_fill_price:,.2f}")
```

### Position Management

```python
from src.order_manager import PositionManager, PositionSide

# Initialize position manager
pm = PositionManager(max_positions=5)

# Open position
position = pm.open_position(
    symbol="BTC/USDT",
    side=PositionSide.LONG,
    entry_price=50000.0,
    quantity=0.1,
    stop_loss=48000.0,
    take_profit=52000.0,
)

# Update price and check PnL
position.update_price(51000.0)
print(f"Unrealized PnL: ${position.unrealized_pnl:.2f}")

# Close position
pm.close_position(
    position_id=position.position_id,
    exit_price=51000.0,
    reason="Manual close"
)
```

### Circuit Breaker

```python
from src.order_manager import CircuitBreaker, CircuitBreakerConfig

# Configure circuit breaker
config = CircuitBreakerConfig(
    max_daily_loss_pct=5.0,        # 5% max daily loss
    max_drawdown_pct=15.0,          # 15% max drawdown
    max_consecutive_losses=5,
    max_orders_per_minute=10,
)

breaker = CircuitBreaker(config)

# Before each trade, check if operational
if not breaker.is_operational:
    print("Trading halted - circuit breaker tripped!")
    return

# After trade, record result
breaker.record_trade(pnl=trade_pnl)

# Check if should trip
breaker.check_and_trip(
    current_pnl=daily_pnl_pct,
    current_drawdown=drawdown_pct
)

# Manual reset after review
breaker.reset(manual=True)
```

### Slippage Simulation (Backtesting)

```python
from src.order_manager import SlippageSimulator, SlippageModel

# Create simulator with realistic model
simulator = SlippageSimulator(model=SlippageModel.REALISTIC)

# Simulate order execution
execution_price, commission = simulator.simulate_execution(
    order=order,
    market_price=50000.0,
    volume_24h=1_000_000_000.0,
    spread_pct=0.02,
)

# Validate order size
is_valid, warning = simulator.validate_order_size(
    order_value=order_value,
    volume_24h=volume_24h,
    max_slippage_pct=0.5
)
```

### Complete Workflow

```python
from src.order_manager import (
    OrderExecutor,
    ExecutionMode,
    CircuitBreaker,
    PositionManager,
    MarketOrder,
    OrderSide,
)

# Initialize components
circuit_breaker = CircuitBreaker()
position_manager = PositionManager(max_positions=3)
executor = OrderExecutor(
    mode=ExecutionMode.BACKTEST,
    circuit_breaker=circuit_breaker,
)

# Check if can trade
if not circuit_breaker.is_operational:
    print("Trading halted!")
    return

# Create and execute order
order = MarketOrder(symbol="BTC/USDT", side=OrderSide.BUY, quantity=0.1)

market_data = {
    "price": 50000.0,
    "volume_24h": 1_000_000_000.0,
}

result = executor.execute(order, market_data=market_data)

if result.success:
    # Open position
    position = position_manager.open_position(
        symbol=order.symbol,
        side=PositionSide.LONG,
        entry_price=result.execution_price,
        quantity=result.filled_quantity,
        stop_loss=result.execution_price * 0.95,
        take_profit=result.execution_price * 1.10,
    )

    print(f"Position opened: {position.position_id}")
```

## Integration with Freqtrade

### In Strategy Class

```python
from freqtrade.strategy import IStrategy
from src.order_manager import CircuitBreaker, PositionManager

class MyStrategy(IStrategy):
    def __init__(self, config: dict):
        super().__init__(config)

        # Initialize order management
        self.circuit_breaker = CircuitBreaker()
        self.position_manager = PositionManager(max_positions=3)

    def populate_entry_trend(self, dataframe, metadata):
        # Check circuit breaker before entry signals
        if not self.circuit_breaker.is_operational:
            dataframe['enter_long'] = 0
            return dataframe

        # ... your entry logic ...

        return dataframe

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        # Check stop-loss via position manager
        position = self.position_manager.get_position_by_trade_id(trade.id)

        if position and position.should_stop_loss:
            return "stop_loss_triggered"

        if position and position.should_take_profit:
            return "take_profit_triggered"

        return None
```

## Testing

Run tests:

```bash
# Run all order management tests
pytest tests/test_order_manager/ -v

# Run specific test file
pytest tests/test_order_manager/test_order_types.py -v
pytest tests/test_order_manager/test_circuit_breaker.py -v

# Run with coverage
pytest tests/test_order_manager/ --cov=src.order_manager --cov-report=html
```

## Examples

Run the comprehensive example script:

```bash
python examples/order_management_example.py
```

This demonstrates:
1. Basic order lifecycle
2. Position management with PnL tracking
3. Circuit breaker protection
4. Slippage simulation
5. Complete trading workflow

## Architecture

```
src/order_manager/
├── __init__.py                  # Public API exports
├── order_types.py               # Order classes & state machine
├── position_manager.py          # Position tracking & PnL
├── circuit_breaker.py           # Risk protection
├── slippage_simulator.py        # Execution simulation
└── order_executor.py            # Order execution engine
```

### Order State Machine

```
PENDING → SUBMITTED → OPEN → [PARTIALLY_FILLED] → FILLED
                    ↓
          CANCELLED/REJECTED/EXPIRED/FAILED
```

Terminal states (cannot transition further):
- FILLED
- CANCELLED
- REJECTED
- EXPIRED
- FAILED

### Circuit Breaker States

```
CLOSED (normal) → OPEN (tripped) → [auto/manual reset] → CLOSED
```

## Configuration

### Circuit Breaker Configuration

```python
from src.order_manager import CircuitBreakerConfig

config = CircuitBreakerConfig(
    # Loss limits
    max_daily_loss_pct=5.0,              # 5% max daily loss
    max_drawdown_pct=15.0,               # 15% max drawdown from peak

    # Consecutive losses
    max_consecutive_losses=5,

    # Order rate limiting
    max_orders_per_minute=10,
    max_orders_per_hour=100,

    # Error limits
    max_consecutive_errors=3,

    # Recovery settings
    auto_reset_after_minutes=60,         # Auto-reset after 1 hour
    require_manual_reset=False,          # Require manual intervention
    cooldown_minutes=15,                 # Cool-down after trip
)
```

### Slippage Configuration

```python
from src.order_manager import SlippageConfig

config = SlippageConfig(
    fixed_slippage_pct=0.05,             # 5 basis points
    volume_slippage_factor=0.1,          # Per 1% of volume
    max_volume_pct=5.0,                  # Max order size
    spread_pct=0.02,                     # Typical spread
    market_impact_factor=0.05,           # Impact coefficient

    # Commission tiers
    commission_tiers={
        "maker": 0.001,                   # 0.1%
        "taker": 0.001,                   # 0.1%
    }
)
```

## Best Practices

### 1. Always Use Circuit Breaker

```python
# DON'T
def place_order(symbol, quantity):
    order = MarketOrder(symbol=symbol, quantity=quantity)
    executor.execute(order)

# DO
def place_order(symbol, quantity):
    if not circuit_breaker.is_operational:
        logger.error("Circuit breaker tripped - order rejected")
        return None

    order = MarketOrder(symbol=symbol, quantity=quantity)
    return executor.execute(order)
```

### 2. Track All Positions

```python
# Open position after successful execution
result = executor.execute(order)

if result.success:
    position = position_manager.open_position(
        symbol=order.symbol,
        side=PositionSide.LONG,
        entry_price=result.execution_price,
        quantity=result.filled_quantity,
        entry_commission=result.commission,
    )
```

### 3. Validate Order Sizes

```python
# Before placing large orders
is_valid, warning = slippage_simulator.validate_order_size(
    order_value=order_value,
    volume_24h=volume_24h,
    max_slippage_pct=0.5
)

if not is_valid:
    logger.warning(f"Order size validation failed: {warning}")
    return
```

### 4. Record All Trade Results

```python
# After closing position
circuit_breaker.record_trade(pnl=position.realized_pnl)

# Update balance for drawdown tracking
circuit_breaker.update_balance(
    current=account_balance,
    peak=peak_balance
)
```

## Monitoring

### Get Circuit Breaker Status

```python
status = circuit_breaker.get_status()

print(f"State: {status['state']}")
print(f"Daily PnL: ${status['daily_pnl']:.2f}")
print(f"Consecutive Losses: {status['consecutive_losses']}")
print(f"Current Drawdown: {status['current_drawdown']:.2f}%")
```

### Get Position Manager Statistics

```python
stats = position_manager.get_statistics()

print(f"Open Positions: {stats['open_positions']}")
print(f"Win Rate: {stats['win_rate']:.1f}%")
print(f"Total PnL: ${stats['total_pnl']:.2f}")
```

### Get Executor Statistics

```python
stats = executor.get_statistics()

print(f"Total Executions: {stats['total_executions']}")
print(f"Success Rate: {stats['success_rate']:.1f}%")
```

## Troubleshooting

### Circuit Breaker Keeps Tripping

1. Check configuration - limits may be too strict
2. Review trade history - identify pattern of losses
3. Consider increasing cooldown period
4. Review strategy logic for bugs

### Orders Not Filling (Backtest)

1. Check market data availability
2. Verify slippage simulator configuration
3. Review order validation logic
4. Check if circuit breaker is tripped

### High Slippage in Backtest

1. Reduce order size relative to volume
2. Use LIMIT orders instead of MARKET
3. Adjust slippage model parameters
4. Consider market conditions (low liquidity periods)

## Future Enhancements

Planned improvements:

1. **ML-Based Slippage Prediction** - Learn from historical execution data
2. **Smart Order Routing** - Route to best exchange/venue
3. **Iceberg Orders** - Break large orders into smaller chunks
4. **TWAP/VWAP Execution** - Time/volume-weighted execution
5. **Portfolio-Level Risk** - Correlation-aware position limits

## References

- Order Types: `src/order_manager/order_types.py`
- Position Manager: `src/order_manager/position_manager.py`
- Circuit Breaker: `src/order_manager/circuit_breaker.py`
- Slippage Simulator: `src/order_manager/slippage_simulator.py`
- Order Executor: `src/order_manager/order_executor.py`
- Examples: `examples/order_management_example.py`
- Tests: `tests/test_order_manager/`
