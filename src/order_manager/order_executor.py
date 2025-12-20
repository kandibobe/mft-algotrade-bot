"""
Order Executor
==============

Handles order execution with retry logic and validation.

Features:
- Pre-execution validation
- Retry on transient failures
- Execution logging
- Integration with slippage simulator
- Circuit breaker integration
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, Dict, Any
from enum import Enum
import time

from src.order_manager.order_types import Order, OrderStatus, OrderType, OrderSide
from src.order_manager.slippage_simulator import SlippageSimulator, SlippageModel
from src.order_manager.circuit_breaker import CircuitBreaker
from typing import Protocol, runtime_checkable
from src.utils.logger import get_logger

# Try to import metrics exporter
try:
    from src.monitoring.metrics_exporter import get_exporter
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

logger = get_logger(__name__)


# =============================================================================
# Dependency Injection Interfaces
# =============================================================================

@runtime_checkable
class ISlippageSimulator(Protocol):
    """Interface for slippage simulator dependency."""
    def simulate_execution(
        self,
        order: Order,
        market_price: float,
        volume_24h: Optional[float] = None,
        spread_pct: Optional[float] = None,
    ) -> tuple[float, float]:
        """Simulate order execution with slippage."""
        ...


@runtime_checkable
class ICircuitBreaker(Protocol):
    """Interface for circuit breaker dependency."""
    @property
    def is_operational(self) -> bool:
        """Check if circuit breaker allows trading."""
        ...
    
    def record_order(self) -> None:
        """Record successful order execution."""
        ...
    
    def record_error(self, error: str) -> None:
        """Record error for circuit breaker analysis."""
        ...


class ExecutionMode(Enum):
    """Order execution mode."""
    LIVE = "live"              # Live trading
    PAPER = "paper"            # Paper trading
    BACKTEST = "backtest"      # Backtesting with simulation


@dataclass
class ExecutionResult:
    """Result of order execution."""

    success: bool
    order: Order
    execution_price: Optional[float] = None
    filled_quantity: float = 0.0
    commission: float = 0.0
    latency_ms: float = 0.0
    error_message: Optional[str] = None
    retry_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "order_id": self.order.order_id,
            "symbol": self.order.symbol,
            "side": self.order.side.value,
            "order_type": self.order.order_type.value,
            "execution_price": self.execution_price,
            "filled_quantity": self.filled_quantity,
            "commission": self.commission,
            "latency_ms": self.latency_ms,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "timestamp": self.timestamp.isoformat(),
        }


class OrderExecutor:
    """
    Executes orders with validation, retry logic, and monitoring.

    Usage:
        executor = OrderExecutor(mode=ExecutionMode.LIVE)

        # Execute order
        result = executor.execute(order, exchange_api)

        if result.success:
            print(f"Order filled @ {result.execution_price}")
        else:
            print(f"Execution failed: {result.error_message}")
    """

    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.LIVE,
        circuit_breaker: Optional[ICircuitBreaker] = None,
        slippage_simulator: Optional[ISlippageSimulator] = None,
        max_retries: int = 3,
        retry_delay_ms: int = 100,
    ):
        """
        Initialize order executor with dependency injection.

        Args:
            mode: Execution mode (live, paper, backtest)
            circuit_breaker: Circuit breaker instance (implements ICircuitBreaker)
            slippage_simulator: Slippage simulator for backtesting (implements ISlippageSimulator)
            max_retries: Maximum retry attempts
            retry_delay_ms: Delay between retries (milliseconds)
        
        Raises:
            ValueError: If required dependencies are not provided for the mode
        """
        self.mode = mode
        
        # Validate and set dependencies
        if slippage_simulator is None and mode in [ExecutionMode.BACKTEST, ExecutionMode.PAPER]:
            # Create default slippage simulator only if needed
            slippage_simulator = SlippageSimulator(model=SlippageModel.REALISTIC)
            logger.info("slippage_simulator_created",
                mode=mode.value,
                model=SlippageModel.REALISTIC.value,
                reason="default_for_backtest_paper"
            )
        
        self.circuit_breaker = circuit_breaker
        self.slippage_simulator = slippage_simulator
        self.max_retries = max_retries
        self.retry_delay_ms = retry_delay_ms

        self.execution_history: list[ExecutionResult] = []
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0

        # Validate dependencies implement required interfaces
        if self.slippage_simulator and not isinstance(self.slippage_simulator, ISlippageSimulator):
            logger.warning("interface_validation_failed",
                component="slippage_simulator",
                interface="ISlippageSimulator"
            )
        
        if self.circuit_breaker and not isinstance(self.circuit_breaker, ICircuitBreaker):
            logger.warning("interface_validation_failed",
                component="circuit_breaker",
                interface="ICircuitBreaker"
            )

        logger.info("order_executor_initialized",
            mode=mode.value,
            max_retries=max_retries,
            retry_delay_ms=retry_delay_ms,
            has_circuit_breaker=self.circuit_breaker is not None,
            has_slippage_simulator=self.slippage_simulator is not None
        )

    def execute(
        self,
        order: Order,
        exchange_api: Optional[Any] = None,
        market_data: Optional[Dict] = None,
    ) -> ExecutionResult:
        """
        Execute an order.

        Args:
            order: Order to execute
            exchange_api: Exchange API instance (for live/paper mode)
            market_data: Current market data (price, volume, etc.)

        Returns:
            ExecutionResult with execution details
        """
        start_time = time.time()
        self.total_executions += 1

        # Check circuit breaker
        if self.circuit_breaker and not self.circuit_breaker.is_operational:
            return self._create_failure_result(
                order, "Circuit breaker is OPEN - trading halted", start_time
            )

        # Pre-execution validation
        is_valid, validation_error = self._validate_order(order, market_data)
        if not is_valid:
            return self._create_failure_result(order, validation_error, start_time)

        # Record order with circuit breaker
        if self.circuit_breaker:
            self.circuit_breaker.record_order()

        # Execute based on mode
        if self.mode == ExecutionMode.BACKTEST:
            result = self._execute_backtest(order, market_data, start_time)
        elif self.mode == ExecutionMode.PAPER:
            result = self._execute_paper(order, exchange_api, market_data, start_time)
        else:  # LIVE
            result = self._execute_live(order, exchange_api, start_time)

        # Update statistics
        if result.success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1
            # Record error with circuit breaker
            if self.circuit_breaker:
                self.circuit_breaker.record_error(result.error_message or "Unknown error")

        # Store execution history
        self.execution_history.append(result)

        # Log execution
        self._log_execution(result)

        # Record metrics if available
        if METRICS_AVAILABLE:
            try:
                exporter = get_exporter()
                if result.success:
                    exporter.record_trade(
                        side=result.order.side.value,
                        status="filled",
                        execution_time=result.latency_ms / 1000.0
                    )
                    exporter.record_order(
                        order_type=result.order.order_type.value,
                        filled=True
                    )
                else:
                    exporter.record_trade(
                        side=result.order.side.value,
                        status="failed",
                        execution_time=result.latency_ms / 1000.0
                    )
                    exporter.record_order(
                        order_type=result.order.order_type.value,
                        filled=False
                    )
            except Exception as e:
                logger.warning("metrics_recording_failed",
                    error=str(e),
                    component="metrics_exporter"
                )

        return result

    def _execute_backtest(
        self,
        order: Order,
        market_data: Optional[Dict],
        start_time: float
    ) -> ExecutionResult:
        """Execute order in backtest mode with slippage simulation."""
        if not market_data:
            return self._create_failure_result(
                order, "Market data required for backtest mode", start_time
            )

        market_price = market_data.get("close", market_data.get("price"))
        if not market_price:
            return self._create_failure_result(
                order, "No market price in market data", start_time
            )

        # Simulate execution with slippage
        try:
            execution_price, commission = self.slippage_simulator.simulate_execution(
                order=order,
                market_price=market_price,
                volume_24h=market_data.get("volume_24h"),
                spread_pct=market_data.get("spread_pct"),
            )

            # Update order
            order.update_status(OrderStatus.SUBMITTED)
            order.update_status(OrderStatus.FILLED)
            order.update_fill(order.quantity, execution_price, commission)

            latency_ms = (time.time() - start_time) * 1000

            return ExecutionResult(
                success=True,
                order=order,
                execution_price=execution_price,
                filled_quantity=order.quantity,
                commission=commission,
                latency_ms=latency_ms,
            )

        except Exception as e:
            return self._create_failure_result(
                order, f"Backtest execution error: {str(e)}", start_time
            )

    def _execute_paper(
        self,
        order: Order,
        exchange_api: Optional[Any],
        market_data: Optional[Dict],
        start_time: float
    ) -> ExecutionResult:
        """
        Execute order in paper trading mode.

        Simulates execution using real market data without actual orders.
        """
        if not market_data:
            return self._create_failure_result(
                order, "Market data required for paper mode", start_time
            )

        market_price = market_data.get("close", market_data.get("price"))

        # Simulate execution (similar to backtest but with slight differences)
        try:
            execution_price, commission = self.slippage_simulator.simulate_execution(
                order=order,
                market_price=market_price,
                volume_24h=market_data.get("volume_24h"),
                spread_pct=market_data.get("spread_pct"),
            )

            # Update order
            order.update_status(OrderStatus.SUBMITTED)
            order.update_status(OrderStatus.FILLED)
            order.update_fill(order.quantity, execution_price, commission)

            latency_ms = (time.time() - start_time) * 1000

            logger.info("paper_order_executed",
                symbol=order.symbol,
                side=order.side.value,
                quantity=order.quantity,
                execution_price=execution_price,
                commission=commission,
                latency_ms=latency_ms,
                mode="paper"
            )

            return ExecutionResult(
                success=True,
                order=order,
                execution_price=execution_price,
                filled_quantity=order.quantity,
                commission=commission,
                latency_ms=latency_ms,
            )

        except Exception as e:
            return self._create_failure_result(
                order, f"Paper execution error: {str(e)}", start_time
            )

    def _execute_live(
        self,
        order: Order,
        exchange_api: Any,
        start_time: float
    ) -> ExecutionResult:
        """
        Execute order on live exchange.

        Implements retry logic for transient failures.
        """
        if not exchange_api:
            return self._create_failure_result(
                order, "Exchange API required for live mode", start_time
            )

        retry_count = 0

        while retry_count <= self.max_retries:
            try:
                # Update order status
                order.update_status(OrderStatus.SUBMITTED)

                # Place order via exchange API
                exchange_result = self._place_order_on_exchange(order, exchange_api)

                if exchange_result["success"]:
                    # Order placed successfully
                    order.exchange_order_id = exchange_result.get("order_id")
                    order.update_status(OrderStatus.OPEN)

                    # Poll for fill (simplified - in production use websocket)
                    fill_result = self._wait_for_fill(order, exchange_api)

                    if fill_result["filled"]:
                        order.update_fill(
                            filled_qty=fill_result["filled_quantity"],
                            fill_price=fill_result["average_price"],
                            commission=fill_result["commission"],
                        )

                        latency_ms = (time.time() - start_time) * 1000

                        return ExecutionResult(
                            success=True,
                            order=order,
                            execution_price=fill_result["average_price"],
                            filled_quantity=fill_result["filled_quantity"],
                            commission=fill_result["commission"],
                            latency_ms=latency_ms,
                            retry_count=retry_count,
                        )
                    else:
                        # Timeout or partial fill
                        error = f"Order not filled within timeout"
                        order.update_status(OrderStatus.EXPIRED, error)
                        return self._create_failure_result(order, error, start_time)

                else:
                    # Order rejected
                    error = exchange_result.get("error", "Unknown error")
                    order.update_status(OrderStatus.REJECTED, error)

                    # Check if should retry
                    if self._is_retryable_error(error) and order.can_retry():
                        retry_count += 1
                        order.increment_retry()

                        # Exponential backoff: delay * (2 ^ retry_count)
                        delay_seconds = (self.retry_delay_ms / 1000) * (2 ** (retry_count - 1))
                        delay_seconds = min(delay_seconds, 30.0)  # Cap at 30 seconds

                        time.sleep(delay_seconds)
                        logger.warning("order_retry",
                            order_id=order.order_id,
                            attempt=retry_count,
                            max_attempts=self.max_retries,
                            delay_seconds=delay_seconds,
                            reason="transient_error"
                        )
                        continue
                    else:
                        return self._create_failure_result(order, error, start_time)

            except Exception as e:
                error = f"Execution exception: {str(e)}"
                logger.error("execution_exception",
                    error_type="execution",
                    message=str(e),
                    order_id=order.order_id,
                    symbol=order.symbol,
                    retry_count=retry_count
                )

                if order.can_retry():
                    retry_count += 1
                    order.increment_retry()

                    # Exponential backoff for exceptions
                    delay_seconds = (self.retry_delay_ms / 1000) * (2 ** (retry_count - 1))
                    delay_seconds = min(delay_seconds, 30.0)  # Cap at 30 seconds

                    logger.warning("exception_retry",
                        order_id=order.order_id,
                        attempt=retry_count,
                        max_attempts=self.max_retries,
                        delay_seconds=delay_seconds,
                        reason="exception"
                    )
                    time.sleep(delay_seconds)
                    continue
                else:
                    order.update_status(OrderStatus.FAILED, error)
                    return self._create_failure_result(order, error, start_time)

        # Max retries exceeded
        error = f"Max retries ({self.max_retries}) exceeded"
        order.update_status(OrderStatus.FAILED, error)
        return self._create_failure_result(order, error, start_time)

    def _place_order_on_exchange(self, order: Order, exchange_api: Any) -> Dict:
        """
        Place order on exchange via API.

        This is a placeholder - actual implementation depends on exchange.
        """
        # Example for CCXT-style API:
        try:
            if order.order_type == OrderType.MARKET:
                result = exchange_api.create_market_order(
                    symbol=order.symbol,
                    side=order.side.value,
                    amount=order.quantity,
                )
            elif order.order_type == OrderType.LIMIT:
                result = exchange_api.create_limit_order(
                    symbol=order.symbol,
                    side=order.side.value,
                    amount=order.quantity,
                    price=order.price,
                )
            else:
                return {"success": False, "error": f"Unsupported order type: {order.order_type}"}

            return {
                "success": True,
                "order_id": result.get("id"),
                "status": result.get("status"),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def _wait_for_fill(self, order: Order, exchange_api: Any, timeout_sec: int = 30) -> Dict:
        """
        Wait for order to fill.

        In production, use websocket instead of polling.
        """
        # Placeholder - implement with exchange API
        # Should poll order status until filled or timeout

        try:
            # Fetch order status
            result = exchange_api.fetch_order(order.exchange_order_id, order.symbol)

            return {
                "filled": result["status"] == "closed",
                "filled_quantity": result.get("filled", 0),
                "average_price": result.get("average", 0),
                "commission": result.get("fee", {}).get("cost", 0),
            }

        except Exception as e:
            logger.error("order_status_fetch_error",
                error_type="connection",
                message=str(e),
                order_id=order.exchange_order_id,
                symbol=order.symbol
            )
            return {"filled": False}

    def _validate_order(self, order: Order, market_data: Optional[Dict]) -> tuple[bool, Optional[str]]:
        """
        Validate order before execution.

        Returns:
            (is_valid, error_message)
        """
        # Basic validation
        if not order.symbol:
            return False, "Order symbol is required"

        if order.quantity <= 0:
            return False, f"Invalid order quantity: {order.quantity}"

        if order.order_type == OrderType.LIMIT and order.price is None:
            return False, "Limit order requires price"

        if order.order_type in [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT] and order.stop_price is None:
            return False, f"{order.order_type.value} requires stop_price"

        # Market data validation (for backtest/paper)
        if self.mode in [ExecutionMode.BACKTEST, ExecutionMode.PAPER]:
            if not market_data:
                return False, "Market data required for backtest/paper mode"

        return True, None

    def _is_retryable_error(self, error: str) -> bool:
        """Check if error is transient and can be retried."""
        retryable_errors = [
            "timeout",
            "connection",
            "network",
            "rate limit",
            "too many requests",
            "service unavailable",
        ]

        error_lower = error.lower()
        return any(retryable in error_lower for retryable in retryable_errors)

    def _create_failure_result(self, order: Order, error: str, start_time: float) -> ExecutionResult:
        """Create execution result for failed order."""
        latency_ms = (time.time() - start_time) * 1000

        return ExecutionResult(
            success=False,
            order=order,
            error_message=error,
            latency_ms=latency_ms,
        )

    def _log_execution(self, result: ExecutionResult):
        """Log execution result."""
        if result.success:
            logger.info("order_execution_success",
                order_id=result.order.order_id,
                symbol=result.order.symbol,
                side=result.order.side.value,
                filled_quantity=result.filled_quantity,
                execution_price=result.execution_price,
                commission=result.commission,
                latency_ms=result.latency_ms,
                order_type=result.order.order_type.value,
                mode=self.mode.value
            )
        else:
            logger.error("order_execution_failed",
                order_id=result.order.order_id,
                symbol=result.order.symbol,
                side=result.order.side.value,
                error_message=result.error_message,
                retry_count=result.retry_count,
                order_type=result.order.order_type.value,
                mode=self.mode.value
            )

    def get_statistics(self) -> Dict:
        """Get execution statistics."""
        success_rate = (
            (self.successful_executions / max(1, self.total_executions)) * 100
        )

        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": success_rate,
            "mode": self.mode.value,
        }
