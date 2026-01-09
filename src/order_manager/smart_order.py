"""
Smart Order Logic
=================

Advanced order types for MFT execution.
"""

import logging
from dataclasses import dataclass

from src.order_manager.order_types import LimitOrder, Order, OrderType

logger = logging.getLogger(__name__)


@dataclass
class SmartOrder(Order):
    """Base class for smart orders."""

    attribution_metadata: dict | None = None
    
    # Latency Tracking
    signal_timestamp: float | None = None
    submission_timestamp: float | None = None
    fill_timestamp: float | None = None

    def on_ticker_update(self, ticker: dict):
        """Handle ticker update to adjust order parameters."""
        pass


@dataclass
class ChaseLimitOrder(LimitOrder, SmartOrder):
    """
    Limit order that chases the best bid/ask.

    Logic:
    - If Buy: Place at Best Bid + Tick Size (to be first in line)
    - If Sell: Place at Best Ask - Tick Size
    - Max Chase Price: Limit price set by user (never buy higher than this)
    """

    max_chase_price: float | None = None  # Worst acceptable price
    chase_offset: float = 0.0  # Offset from best bid/ask

    def __post_init__(self):
        super().__post_init__()
        if self.max_chase_price is None:
            self.max_chase_price = self.price  # Default to initial price as limit

    def on_ticker_update(self, ticker: dict):
        """
        Adjust price based on new ticker and L2 imbalance.
        
        MFT Optimization:
        - If Buy + Positive Imbalance: stay at Best Bid (passive).
        - If Buy + Negative Imbalance: move closer to Best Ask or increase offset (aggressive).
        """
        if not self.is_active:
            return

        best_bid = ticker.get("best_bid")
        best_ask = ticker.get("best_ask")
        imbalance = ticker.get("imbalance", 0.0)

        if not best_bid or not best_ask:
            return

        new_price = self.price
        
        # Adaptive offset based on imbalance
        # imbalance > 0 means more bids (buying pressure)
        # imbalance < 0 means more asks (selling pressure)
        dynamic_offset = self.chase_offset
        
        if self.is_buy:
            if imbalance < -0.3: # Selling pressure, price might drop or we might get front-run
                dynamic_offset += 0.00001 # Micro-increase to be first in line
            
            target_price = best_bid + dynamic_offset
            new_price = min(target_price, self.max_chase_price)

        else: # Sell
            if imbalance > 0.3: # Buying pressure, price might rise
                dynamic_offset += 0.00001
                
            target_price = best_ask - dynamic_offset
            new_price = max(target_price, self.max_chase_price)

        # If price changed significantly, update
        if abs(new_price - self.price) > 0.0000001:
            logger.info(
                f"SmartOrder {self.order_id}: Adjusting price {self.price} -> {new_price} "
                f"(Imbalance: {imbalance:.2f})"
            )
            self.price = new_price


@dataclass
class TWAPOrder(SmartOrder):
    """
    Time-Weighted Average Price order.
    Splits the order into smaller chunks to be executed over a set duration.
    """

    duration_minutes: int = 60
    num_chunks: int = 12

    def __post_init__(self):
        super().__post_init__()
        self.order_type = OrderType.TWAP


@dataclass
class VWAPOrder(SmartOrder):
    """
    Volume-Weighted Average Price order.
    Splits the order into smaller chunks based on historical volume profile.
    """

    duration_minutes: int = 60
    num_chunks: int = 12
    volume_profile: list[float] | None = None  # List of volume percentages for each chunk

    def __post_init__(self):
        super().__post_init__()
        self.order_type = OrderType.VWAP

@dataclass
class PeggedOrder(SmartOrder):
    """
    Order pegged to a reference price (Best Bid/Ask).
    
    Logic:
    - Automatically updates price as market moves.
    - Maintains 'offset' distance.
    - Primary Peg: Buy @ Best Bid / Sell @ Best Ask
    - Opposite Peg: Buy @ Best Ask / Sell @ Best Bid (Aggressive)
    """
    
    offset: float = 0.0
    peg_side: str = "primary" # 'primary' (same side) or 'opposite' (crossing spread)
    
    def __post_init__(self):
        super().__post_init__()
        self.order_type = OrderType.PEGGED
        
    def on_ticker_update(self, ticker: dict):
        """Adjust price based on pegged reference."""
        if not self.is_active:
            return

        best_bid = ticker.get("best_bid")
        best_ask = ticker.get("best_ask")
        
        if not best_bid or not best_ask:
            return
            
        new_price = self.price
        
        if self.is_buy:
            reference = best_bid if self.peg_side == "primary" else best_ask
            new_price = reference + self.offset
        else: # Sell
            reference = best_ask if self.peg_side == "primary" else best_bid
            new_price = reference - self.offset
            
        # Update if changed significantly
        if abs(new_price - self.price) > 0.0000001:
            logger.info(f"PeggedOrder {self.order_id}: Adjusting price {self.price} -> {new_price} (Ref: {reference})")
            self.price = new_price
