"""
Stoic Citadel - Liquidation Guard
=================================

Protects against liquidation risk by ensuring stop-losses are safe
relative to estimated liquidation prices.

Logic:
- Calculate estimated liquidation price based on leverage and margin mode.
- Ensure Stop Loss is triggered BEFORE liquidation price (with buffer).
- Reject trades where liquidation risk is too high.

Formulas:
    Long Liq Price = Entry * (1 - 1/Leverage + MMR)
    Short Liq Price = Entry * (1 + 1/Leverage - MMR)
    (Simplified approximation, varies by exchange)

Author: Stoic Citadel Team
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LiquidationConfig:
    """Configuration for liquidation guard."""

    # ADL Protection
    max_adl_score: int = 3 # Trigger reduction if ADL score >= 3 (out of 5)

    # Safety buffer: Stop loss must be this much closer to entry than liq price
    # e.g. 0.20 means SL must be at 80% of the distance to liquidation
    safety_buffer: float = 0.20

    # Default Maintenance Margin Rate (MMR) if not provided
    # Binance usually starts at 0.4% or 0.5% for major pairs
    default_mmr: float = 0.005

    # Max allowed leverage before warning/blocking
    max_safe_leverage: float = 5.0


class LiquidationGuard:
    """
    Guard against liquidation events.

    Calculates liquidation prices and validates trade safety.
    """

    def __init__(self, config: LiquidationConfig | None = None):
        self.config = config or LiquidationConfig()

    def check_adl_risk(self, adl_score: int) -> tuple[bool, str]:
        """
        Check if Auto-Deleverage risk is too high.
        
        Args:
            adl_score: 0-5 scale of ADL risk.
            
        Returns:
            (is_safe, reason)
        """
        if adl_score >= self.config.max_adl_score:
            return False, f"ADL Risk too high: {adl_score}/5"
        return True, "ADL Risk safe"

    def calculate_liquidation_price(
        self,
        entry_price: float,
        leverage: float,
        side: str,
        wallet_balance: float | None = None,
        position_size: float | None = None,
        mmr: float | None = None,
    ) -> float:
        """
        Calculate estimated liquidation price (Isolated Margin).

        Args:
            entry_price: Entry price
            leverage: Leverage used (e.g. 3.0)
            side: 'long' or 'short'
            mmr: Maintenance Margin Rate (default from config)

        Returns:
            Estimated liquidation price
        """
        if leverage <= 1.0:
            return 0.0 if side == "long" else float("inf")

        mmr = mmr or self.config.default_mmr

        # Simplified isolated margin formula
        # Initial Margin = Entry / Leverage
        # Maintenance Margin = Entry * MMR
        # Liquidation when: Collateral - Loss <= Maintenance Margin
        # Loss = (Entry - Current) for Long

        # For Long:
        # Initial Margin - (Entry - Liq) = Liq * MMR
        # (Entry/Lev) - Entry + Liq = Liq * MMR
        # Liq * (1 - MMR) = Entry * (1 - 1/Lev)
        # Liq = Entry * (1 - 1/Lev) / (1 - MMR)

        if side == "long":
            liq_price = entry_price * (1 - 1 / leverage) / (1 - mmr)
            return max(0, liq_price)
        else:
            # For Short:
            # Initial Margin - (Liq - Entry) = Liq * MMR
            # (Entry/Lev) - Liq + Entry = Liq * MMR
            # Entry * (1 + 1/Lev) = Liq * (1 + MMR)
            # Liq = Entry * (1 + 1/Lev) / (1 + MMR)
            liq_price = entry_price * (1 + 1 / leverage) / (1 + mmr)
            return liq_price

    def check_trade_safety(
        self, entry_price: float, stop_loss_price: float, leverage: float, side: str
    ) -> tuple[bool, str]:
        """
        Check if a trade is safe from liquidation.

        Args:
            entry_price: Planned entry price
            stop_loss_price: Planned stop loss price
            leverage: Leverage to use
            side: 'long' or 'short'

        Returns:
            (is_safe, reason)
        """
        if leverage <= 1.0:
            return True, "Safe (No leverage)"

        liq_price = self.calculate_liquidation_price(entry_price, leverage, side)

        dist_to_liq = abs(entry_price - liq_price)
        dist_to_sl = abs(entry_price - stop_loss_price)

        # Check if SL is beyond Liquidation (Instant Kill)
        if side == "long":
            if stop_loss_price <= liq_price:
                return (
                    False,
                    f"Stop Loss ({stop_loss_price:.2f}) is below Liquidation Price ({liq_price:.2f})",
                )
        else:
            if stop_loss_price >= liq_price:
                return (
                    False,
                    f"Stop Loss ({stop_loss_price:.2f}) is above Liquidation Price ({liq_price:.2f})",
                )

        # Check Safety Buffer
        # We want SL to be triggered well before Liq
        # dist_to_sl should be < dist_to_liq * (1 - safety_buffer)

        max_safe_dist = dist_to_liq * (1 - self.config.safety_buffer)

        if dist_to_sl > max_safe_dist:
            recommended_sl = (
                entry_price - max_safe_dist if side == "long" else entry_price + max_safe_dist
            )
            return False, (
                f"Liquidation Risk: Stop Loss is too close to Liquidation Price. "
                f"Liq: {liq_price:.2f}, SL: {stop_loss_price:.2f}. "
                f"Max Safe Dist: {max_safe_dist:.2f} (Buffer: {self.config.safety_buffer:.0%}). "
                f"Recommended SL: {recommended_sl:.2f}"
            )

        return True, f"Safe. Liq: {liq_price:.2f} (Dist: {dist_to_liq:.2f})"

    def get_max_safe_leverage(self, entry_price: float, stop_loss_price: float, side: str) -> float:
        """
        Calculate maximum safe leverage for a given stop loss.

        Inverse of check_trade_safety logic.
        """
        dist_to_sl = abs(entry_price - stop_loss_price)

        # We want: dist_to_sl <= dist_to_liq * (1 - buffer)
        # So: dist_to_liq >= dist_to_sl / (1 - buffer)

        required_dist_to_liq = dist_to_sl / (1 - self.config.safety_buffer)

        # For Long: Liq = Entry - ReqDist
        # Liq = Entry * (1 - 1/Lev) / (1 - MMR)
        # Entry - ReqDist = Entry * (1 - 1/Lev) / (1 - MMR)
        # (1 - ReqDist/Entry) * (1 - MMR) = 1 - 1/Lev
        # 1/Lev = 1 - (1 - ReqDist/Entry) * (1 - MMR)

        mmr = self.config.default_mmr

        if side == "long":
            target_liq = entry_price - required_dist_to_liq
            if target_liq <= 0:
                return 1.0

            # Solve for Lev
            term = (target_liq / entry_price) * (1 - mmr)
            inv_lev = 1 - term

            if inv_lev <= 0:
                return 100.0  # High leverage possible
            return 1.0 / inv_lev

        else:
            target_liq = entry_price + required_dist_to_liq

            # Solve for Lev
            # Liq = Entry * (1 + 1/Lev) / (1 + MMR)
            # target_liq / Entry * (1 + MMR) = 1 + 1/Lev
            # 1/Lev = (target_liq / Entry * (1 + MMR)) - 1

            term = (target_liq / entry_price) * (1 + mmr)
            inv_lev = term - 1

            if inv_lev <= 0:
                return 100.0
            return 1.0 / inv_lev
