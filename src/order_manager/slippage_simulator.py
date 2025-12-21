"""
Slippage Simulator
==================

Realistic order execution simulation for backtesting.

Models:
- Volume-based slippage
- Market impact
- Spread costs
- Commission tiers
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from src.order_manager.order_types import Order, OrderSide, OrderType

logger = logging.getLogger(__name__)


class SlippageModel(Enum):
    """Slippage modeling approach."""

    FIXED = "fixed"  # Fixed percentage
    VOLUME_BASED = "volume_based"  # Based on order size vs volume
    SPREAD_BASED = "spread_based"  # Based on bid-ask spread
    REALISTIC = "realistic"  # Combines multiple factors


@dataclass
class SlippageConfig:
    """Configuration for slippage simulation."""

    # Fixed slippage
    fixed_slippage_pct: float = 0.05  # 0.05% = 5 basis points

    # Volume-based slippage
    volume_slippage_factor: float = 0.1  # Slippage per 1% of volume
    max_volume_pct: float = 5.0  # Max order size as % of volume

    # Spread-based
    spread_pct: float = 0.02  # Typical bid-ask spread %

    # Market impact
    market_impact_factor: float = 0.05  # Impact coefficient

    # Commission tiers (volume-based)
    commission_tiers: dict = None

    def __post_init__(self):
        """Set default commission tiers."""
        if self.commission_tiers is None:
            # Default commission structure (Binance-like)
            self.commission_tiers = {
                "maker": 0.001,  # 0.1%
                "taker": 0.001,  # 0.1%
                "vip1": 0.0009,
                "vip2": 0.0008,
            }


class SlippageSimulator:
    """
    Simulates realistic order execution for backtesting.

    Ensures backtest results are more accurate by modeling:
    - Price slippage based on order size
    - Market impact
    - Bid-ask spread
    - Commission costs
    """

    def __init__(
        self,
        model: SlippageModel = SlippageModel.REALISTIC,
        config: Optional[SlippageConfig] = None,
    ):
        """
        Initialize slippage simulator.

        Args:
            model: Slippage model to use
            config: Slippage configuration
        """
        self.model = model
        self.config = config or SlippageConfig()

        logger.info(f"Slippage simulator initialized with {model.value} model")

    def simulate_execution(
        self,
        order: Order,
        market_price: float,
        volume_24h: Optional[float] = None,
        spread_pct: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Simulate order execution with slippage.

        Args:
            order: Order to execute
            market_price: Current market price
            volume_24h: 24h trading volume (for volume-based model)
            spread_pct: Current bid-ask spread (optional)

        Returns:
            (execution_price, commission)
        """
        if self.model == SlippageModel.FIXED:
            execution_price, commission = self._simulate_fixed(order, market_price)
        elif self.model == SlippageModel.VOLUME_BASED:
            execution_price, commission = self._simulate_volume_based(
                order, market_price, volume_24h
            )
        elif self.model == SlippageModel.SPREAD_BASED:
            execution_price, commission = self._simulate_spread_based(
                order, market_price, spread_pct
            )
        else:  # REALISTIC
            execution_price, commission = self._simulate_realistic(
                order, market_price, volume_24h, spread_pct
            )

        # Log execution
        slippage_pct = abs(execution_price - market_price) / market_price * 100
        logger.debug(
            f"Order {order.order_id} simulated: "
            f"Market: {market_price:.2f} → Fill: {execution_price:.2f} "
            f"(Slippage: {slippage_pct:.3f}%, Commission: {commission:.4f})"
        )

        return execution_price, commission

    def _simulate_fixed(self, order: Order, market_price: float) -> Tuple[float, float]:
        """Simulate with fixed slippage percentage."""
        slippage_pct = self.config.fixed_slippage_pct / 100

        if order.side == OrderSide.BUY:
            # Buy at slightly higher price
            execution_price = market_price * (1 + slippage_pct)
        else:
            # Sell at slightly lower price
            execution_price = market_price * (1 - slippage_pct)

        commission = self._calculate_commission(order, execution_price)

        return execution_price, commission

    def _simulate_volume_based(
        self, order: Order, market_price: float, volume_24h: Optional[float]
    ) -> Tuple[float, float]:
        """Simulate slippage based on order size vs market volume."""
        if volume_24h is None or volume_24h == 0:
            # Fall back to fixed slippage
            return self._simulate_fixed(order, market_price)

        # Calculate order value
        order_value = order.quantity * market_price

        # Order size as percentage of 24h volume
        order_pct = (order_value / volume_24h) * 100

        # Cap at max volume percentage
        order_pct = min(order_pct, self.config.max_volume_pct)

        # Slippage increases with order size
        slippage_pct = order_pct * self.config.volume_slippage_factor / 100

        if order.side == OrderSide.BUY:
            execution_price = market_price * (1 + slippage_pct)
        else:
            execution_price = market_price * (1 - slippage_pct)

        commission = self._calculate_commission(order, execution_price)

        return execution_price, commission

    def _simulate_spread_based(
        self, order: Order, market_price: float, spread_pct: Optional[float]
    ) -> Tuple[float, float]:
        """Simulate slippage based on bid-ask spread."""
        spread = spread_pct if spread_pct else self.config.spread_pct

        if order.side == OrderSide.BUY:
            # Buy at ask price (mid + half spread)
            execution_price = market_price * (1 + spread / 200)
        else:
            # Sell at bid price (mid - half spread)
            execution_price = market_price * (1 - spread / 200)

        commission = self._calculate_commission(order, execution_price)

        return execution_price, commission

    def _simulate_realistic(
        self,
        order: Order,
        market_price: float,
        volume_24h: Optional[float],
        spread_pct: Optional[float],
    ) -> Tuple[float, float]:
        """
        Realistic simulation combining multiple factors.

        Components:
        1. Bid-ask spread cost
        2. Market impact (price moves against you)
        3. Random microstructure noise
        """
        # 1. Spread cost
        spread = spread_pct if spread_pct else self.config.spread_pct
        spread_cost = spread / 200  # Half spread

        # 2. Market impact
        market_impact = 0.0
        if volume_24h:
            order_value = order.quantity * market_price
            order_pct = (order_value / volume_24h) * 100
            market_impact = order_pct * self.config.market_impact_factor / 100

        # 3. Random microstructure noise (small random component)
        noise = np.random.normal(0, 0.0002)  # ~0.02% std dev

        # Combine factors
        total_slippage = spread_cost + market_impact + noise

        if order.side == OrderSide.BUY:
            execution_price = market_price * (1 + total_slippage)
        else:
            execution_price = market_price * (1 - total_slippage)

        # Commission (taker fee for market orders, maker for limit orders)
        if order.order_type == OrderType.MARKET:
            commission_rate = self.config.commission_tiers["taker"]
        else:
            commission_rate = self.config.commission_tiers["maker"]

        commission = order.quantity * execution_price * commission_rate

        return execution_price, commission

    def _calculate_commission(self, order: Order, execution_price: float) -> float:
        """
        Calculate commission based on order type and tier.

        Args:
            order: Order object
            execution_price: Execution price

        Returns:
            Commission amount
        """
        # Market orders pay taker fee, limit orders pay maker fee
        if order.order_type == OrderType.MARKET:
            rate = self.config.commission_tiers["taker"]
        else:
            rate = self.config.commission_tiers["maker"]

        commission = order.quantity * execution_price * rate

        return commission

    def estimate_worst_case_slippage(
        self, order_value: float, market_price: float, volume_24h: float
    ) -> float:
        """
        Estimate worst-case slippage percentage.

        Useful for risk assessment before placing large orders.

        Args:
            order_value: Order value in quote currency
            market_price: Current market price
            volume_24h: 24h trading volume

        Returns:
            Worst-case slippage percentage
        """
        order_pct = (order_value / volume_24h) * 100

        # Worst case: spread + market impact
        spread_cost = self.config.spread_pct / 2
        market_impact = order_pct * self.config.market_impact_factor

        worst_case_pct = spread_cost + market_impact

        return worst_case_pct

    def validate_order_size(
        self, order_value: float, volume_24h: float, max_slippage_pct: float = 0.5
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate if order size is reasonable given market conditions.

        Args:
            order_value: Order value
            volume_24h: 24h trading volume
            max_slippage_pct: Maximum acceptable slippage

        Returns:
            (is_valid, warning_message)
        """
        if volume_24h == 0:
            return False, "No market volume data available"

        order_pct = (order_value / volume_24h) * 100

        # Check if order is too large
        if order_pct > self.config.max_volume_pct:
            return False, (
                f"Order too large: {order_pct:.2f}% of 24h volume "
                f"(max: {self.config.max_volume_pct}%)"
            )

        # Estimate slippage
        estimated_slippage = self.estimate_worst_case_slippage(
            order_value, order_value / 1000, volume_24h
        )

        if estimated_slippage > max_slippage_pct:
            return False, (
                f"Estimated slippage too high: {estimated_slippage:.2f}% "
                f"(max: {max_slippage_pct}%)"
            )

        return True, None
