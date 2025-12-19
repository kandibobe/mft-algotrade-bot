"""
Pre-Trade Checks - Order Validation Before Execution
=====================================================

Validates orders before sending to exchange to prevent:
- Insufficient balance errors
- Invalid order sizes (below min notional)
- Excessive leverage
- Violating risk limits
- Invalid prices (too far from market)

CRITICAL: All orders MUST pass pre-trade checks before execution.

Usage:
    checker = PreTradeChecker(config)

    # Before sending order
    result = checker.validate_order(
        symbol='BTC/USDT',
        side='buy',
        quantity=0.001,
        price=50000.0,
        current_balance=10000.0,
        current_price=50500.0
    )

    if not result.passed:
        logger.error(f"Pre-trade check failed: {result.reason}")
        return False
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class CheckResult(Enum):
    """Result of pre-trade check."""
    PASSED = "passed"
    FAILED_BALANCE = "insufficient_balance"
    FAILED_NOTIONAL = "below_min_notional"
    FAILED_LEVERAGE = "excessive_leverage"
    FAILED_PRICE = "price_out_of_range"
    FAILED_SIZE = "invalid_size"
    FAILED_RISK = "risk_limit_exceeded"


@dataclass
class ValidationResult:
    """Result of order validation."""
    passed: bool
    result: CheckResult
    reason: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class PreTradeConfig:
    """Configuration for pre-trade checks."""

    # Balance checks
    min_balance_buffer: float = 0.05  # Keep 5% balance as buffer
    max_balance_per_trade: float = 0.20  # Max 20% of balance per trade

    # Notional checks
    min_notional_usd: float = 10.0  # Min $10 order size
    max_notional_usd: float = 100000.0  # Max $100k per order

    # Leverage checks
    max_leverage: float = 3.0  # Max 3x leverage
    allow_margin: bool = False  # Disable margin trading by default

    # Price checks (deviation from market)
    max_price_deviation_pct: float = 5.0  # Max 5% from current price

    # Size checks
    min_quantity: float = 0.00001  # Min quantity
    max_quantity: float = 1000.0  # Max quantity

    # Risk checks
    max_open_positions: int = 5  # Max simultaneous positions
    max_daily_trades: int = 100  # Max trades per day


class PreTradeChecker:
    """
    Pre-trade order validator.

    Performs comprehensive checks before order execution to prevent errors.

    Conservative defaults for retail traders:
    - Max 20% of balance per trade
    - Max 3x leverage (preferably 1x)
    - Min $10 order size
    - Max 5% price deviation
    - Max 5 open positions
    """

    def __init__(self, config: Optional[PreTradeConfig] = None):
        """
        Initialize pre-trade checker.

        Args:
            config: Configuration (defaults to conservative settings)
        """
        self.config = config or PreTradeConfig()

        logger.info(
            f"Pre-trade checker initialized: "
            f"max_balance_per_trade={self.config.max_balance_per_trade:.0%}, "
            f"max_leverage={self.config.max_leverage}x, "
            f"min_notional=${self.config.min_notional_usd}"
        )

    def validate_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float],
        order_type: str = "limit",
        current_balance: float = 0.0,
        current_price: Optional[float] = None,
        current_positions: int = 0,
        daily_trade_count: int = 0,
    ) -> ValidationResult:
        """
        Validate order against all pre-trade checks.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            side: 'buy' or 'sell'
            quantity: Order quantity
            price: Order price (None for market orders)
            order_type: Order type ('limit', 'market')
            current_balance: Current account balance
            current_price: Current market price (for deviation check)
            current_positions: Number of open positions
            daily_trade_count: Number of trades today

        Returns:
            ValidationResult with pass/fail and details
        """
        # 1. Check quantity
        result = self._check_quantity(quantity)
        if not result.passed:
            return result

        # 2. Check price (if limit order)
        if order_type == "limit" and price is not None and current_price is not None:
            result = self._check_price_deviation(price, current_price, side)
            if not result.passed:
                return result

        # 3. Check notional value
        order_price = price if price is not None else current_price
        if order_price is not None:
            result = self._check_notional(quantity, order_price)
            if not result.passed:
                return result

        # 4. Check balance
        if current_balance > 0 and order_price is not None:
            result = self._check_balance(
                quantity, order_price, current_balance, side
            )
            if not result.passed:
                return result

        # 5. Check position limits
        result = self._check_position_limits(current_positions)
        if not result.passed:
            return result

        # 6. Check daily trade limits
        result = self._check_daily_limits(daily_trade_count)
        if not result.passed:
            return result

        # All checks passed
        logger.info(
            f"✅ Pre-trade checks PASSED for {symbol} {side} {quantity} @ {order_price}"
        )

        return ValidationResult(
            passed=True,
            result=CheckResult.PASSED,
            reason="All pre-trade checks passed"
        )

    def _check_quantity(self, quantity: float) -> ValidationResult:
        """Check if quantity is within allowed range."""
        if quantity < self.config.min_quantity:
            return ValidationResult(
                passed=False,
                result=CheckResult.FAILED_SIZE,
                reason=f"Quantity {quantity} below minimum {self.config.min_quantity}",
                details={"quantity": quantity, "min": self.config.min_quantity}
            )

        if quantity > self.config.max_quantity:
            return ValidationResult(
                passed=False,
                result=CheckResult.FAILED_SIZE,
                reason=f"Quantity {quantity} exceeds maximum {self.config.max_quantity}",
                details={"quantity": quantity, "max": self.config.max_quantity}
            )

        return ValidationResult(
            passed=True,
            result=CheckResult.PASSED,
            reason="Quantity check passed"
        )

    def _check_notional(self, quantity: float, price: float) -> ValidationResult:
        """Check if order notional value is within limits."""
        notional = quantity * price

        if notional < self.config.min_notional_usd:
            return ValidationResult(
                passed=False,
                result=CheckResult.FAILED_NOTIONAL,
                reason=f"Order notional ${notional:.2f} below minimum ${self.config.min_notional_usd}",
                details={
                    "notional": notional,
                    "min_notional": self.config.min_notional_usd,
                    "quantity": quantity,
                    "price": price
                }
            )

        if notional > self.config.max_notional_usd:
            return ValidationResult(
                passed=False,
                result=CheckResult.FAILED_NOTIONAL,
                reason=f"Order notional ${notional:.2f} exceeds maximum ${self.config.max_notional_usd}",
                details={
                    "notional": notional,
                    "max_notional": self.config.max_notional_usd
                }
            )

        return ValidationResult(
            passed=True,
            result=CheckResult.PASSED,
            reason="Notional check passed"
        )

    def _check_balance(
        self,
        quantity: float,
        price: float,
        balance: float,
        side: str
    ) -> ValidationResult:
        """Check if sufficient balance for order."""
        notional = quantity * price
        required_balance = notional * (1 + self.config.min_balance_buffer)

        if side == "buy" and required_balance > balance:
            return ValidationResult(
                passed=False,
                result=CheckResult.FAILED_BALANCE,
                reason=f"Insufficient balance: need ${required_balance:.2f}, have ${balance:.2f}",
                details={
                    "required": required_balance,
                    "available": balance,
                    "shortfall": required_balance - balance
                }
            )

        # Check max per trade
        max_allowed = balance * self.config.max_balance_per_trade
        if notional > max_allowed:
            return ValidationResult(
                passed=False,
                result=CheckResult.FAILED_RISK,
                reason=f"Order ${notional:.2f} exceeds max per trade ${max_allowed:.2f} "
                       f"({self.config.max_balance_per_trade:.0%} of balance)",
                details={
                    "order_size": notional,
                    "max_allowed": max_allowed,
                    "balance": balance
                }
            )

        return ValidationResult(
            passed=True,
            result=CheckResult.PASSED,
            reason="Balance check passed"
        )

    def _check_price_deviation(
        self,
        order_price: float,
        market_price: float,
        side: str
    ) -> ValidationResult:
        """Check if price is within acceptable range of market price."""
        deviation = abs((order_price - market_price) / market_price) * 100

        if deviation > self.config.max_price_deviation_pct:
            return ValidationResult(
                passed=False,
                result=CheckResult.FAILED_PRICE,
                reason=f"Price deviation {deviation:.2f}% exceeds limit {self.config.max_price_deviation_pct:.2f}%",
                details={
                    "order_price": order_price,
                    "market_price": market_price,
                    "deviation_pct": deviation,
                    "max_deviation": self.config.max_price_deviation_pct
                }
            )

        # Additional check: buy orders shouldn't be too high, sell too low
        if side == "buy" and order_price > market_price * 1.02:
            logger.warning(
                f"Buy order price ${order_price:.2f} is above market ${market_price:.2f}"
            )

        if side == "sell" and order_price < market_price * 0.98:
            logger.warning(
                f"Sell order price ${order_price:.2f} is below market ${market_price:.2f}"
            )

        return ValidationResult(
            passed=True,
            result=CheckResult.PASSED,
            reason="Price deviation check passed"
        )

    def _check_position_limits(self, current_positions: int) -> ValidationResult:
        """Check if opening new position would exceed limit."""
        if current_positions >= self.config.max_open_positions:
            return ValidationResult(
                passed=False,
                result=CheckResult.FAILED_RISK,
                reason=f"Max positions limit reached: {current_positions}/{self.config.max_open_positions}",
                details={
                    "current_positions": current_positions,
                    "max_positions": self.config.max_open_positions
                }
            )

        return ValidationResult(
            passed=True,
            result=CheckResult.PASSED,
            reason="Position limit check passed"
        )

    def _check_daily_limits(self, daily_trade_count: int) -> ValidationResult:
        """Check if daily trade limit would be exceeded."""
        if daily_trade_count >= self.config.max_daily_trades:
            return ValidationResult(
                passed=False,
                result=CheckResult.FAILED_RISK,
                reason=f"Daily trade limit reached: {daily_trade_count}/{self.config.max_daily_trades}",
                details={
                    "daily_trades": daily_trade_count,
                    "max_daily_trades": self.config.max_daily_trades
                }
            )

        return ValidationResult(
            passed=True,
            result=CheckResult.PASSED,
            reason="Daily limit check passed"
        )

    def validate_leverage(
        self,
        position_size: float,
        collateral: float
    ) -> ValidationResult:
        """
        Validate leverage is within limits.

        Args:
            position_size: Total position notional value
            collateral: Available collateral

        Returns:
            ValidationResult
        """
        if not self.config.allow_margin:
            return ValidationResult(
                passed=False,
                result=CheckResult.FAILED_LEVERAGE,
                reason="Margin trading is disabled",
                details={"allow_margin": False}
            )

        if collateral <= 0:
            return ValidationResult(
                passed=False,
                result=CheckResult.FAILED_BALANCE,
                reason="No collateral available",
                details={"collateral": collateral}
            )

        leverage = position_size / collateral

        if leverage > self.config.max_leverage:
            return ValidationResult(
                passed=False,
                result=CheckResult.FAILED_LEVERAGE,
                reason=f"Leverage {leverage:.2f}x exceeds maximum {self.config.max_leverage}x",
                details={
                    "leverage": leverage,
                    "max_leverage": self.config.max_leverage,
                    "position_size": position_size,
                    "collateral": collateral
                }
            )

        return ValidationResult(
            passed=True,
            result=CheckResult.PASSED,
            reason="Leverage check passed",
            details={"leverage": leverage}
        )

    def get_max_order_size(
        self,
        balance: float,
        price: float,
        include_buffer: bool = True
    ) -> float:
        """
        Calculate maximum allowed order size.

        Args:
            balance: Available balance
            price: Order price
            include_buffer: Include balance buffer

        Returns:
            Maximum quantity
        """
        max_notional = min(
            balance * self.config.max_balance_per_trade,
            self.config.max_notional_usd
        )

        if include_buffer:
            max_notional *= (1 - self.config.min_balance_buffer)

        max_quantity = max_notional / price

        # Respect max quantity limit
        max_quantity = min(max_quantity, self.config.max_quantity)

        return max_quantity
