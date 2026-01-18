"""
Order Management System
=======================

Comprehensive order management for algorithmic trading:
- Order types (Market, Limit, Stop-Loss, Take-Profit)
- Position tracking and management
- Execution simulation with slippage
- Circuit breaker for risk protection
- Detailed execution logging

Author: Stoic Citadel Team
License: MIT
"""

from src.order_manager.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
)

# from src.order_manager.order_executor import (
#     ExecutionMode,
#     ExecutionResult,
#     OrderExecutor,
# )
from src.order_manager.order_types import (
    LimitOrder,
    MarketOrder,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    StopLossOrder,
    TakeProfitOrder,
    TrailingStopOrder,
)
from src.order_manager.position_manager import (
    Position,
    PositionManager,
    PositionSide,
)
from src.order_manager.slippage_simulator import (
    SlippageModel,
    SlippageSimulator,
)

try:
    from src.order_manager.smart_limit_executor import (
        ChasingStrategy,
        SmartExecutionResult,
        SmartLimitConfig,
        SmartLimitExecutor,
    )
except ImportError:
    ChasingStrategy = None
    SmartExecutionResult = None
    SmartLimitConfig = None
    SmartLimitExecutor = None

from src.order_manager.smart_order_executor import SmartOrderExecutor

__all__ = [
    "ChasingStrategy",
    # Risk management
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "LimitOrder",
    "MarketOrder",
    # Order types
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    # Position management
    "Position",
    "PositionManager",
    "PositionSide",
    "SlippageModel",
    # Simulation
    "SlippageSimulator",
    "SmartExecutionResult",
    "SmartLimitConfig",
    # Smart execution
    "SmartLimitExecutor",
    # Execution
    # "OrderExecutor",
    # "ExecutionResult",
    # "ExecutionMode",
    "SmartOrderExecutor",
    "StopLossOrder",
    "TakeProfitOrder",
    "TrailingStopOrder",
]
