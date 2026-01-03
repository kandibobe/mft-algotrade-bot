"""
Smart Order Logic
=================

Advanced order types for MFT execution.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime

from src.order_manager.order_types import Order, OrderType, OrderStatus, LimitOrder

logger = logging.getLogger(__name__)

@dataclass
class SmartOrder(Order):
    """Base class for smart orders."""
    attribution_metadata: Optional[Dict] = None

    def on_ticker_update(self, ticker: dict):
        """Handle ticker update to adjust order parameters."""
        pass

@dataclass
class ChaseLimitOrder(LimitOrder):
    """
    Limit order that chases the best bid/ask.
    
    Logic:
    - If Buy: Place at Best Bid + Tick Size (to be first in line)
    - If Sell: Place at Best Ask - Tick Size
    - Max Chase Price: Limit price set by user (never buy higher than this)
    """
    
    max_chase_price: Optional[float] = None # Worst acceptable price
    chase_offset: float = 0.0 # Offset from best bid/ask
    
    def __post_init__(self):
        super().__post_init__()
        if self.max_chase_price is None:
            self.max_chase_price = self.price # Default to initial price as limit
            
    def on_ticker_update(self, ticker: dict):
        """Adjust price based on new ticker."""
        if not self.is_active:
            return

        best_bid = ticker.get('best_bid')
        best_ask = ticker.get('best_ask')
        
        if not best_bid or not best_ask:
            return

        new_price = self.price
        
        if self.is_buy:
            # Target: Best Bid (plus offset to be ahead?)
            # For simplicity, match Best Bid
            target_price = best_bid + self.chase_offset
            
            # Don't exceed max price
            if target_price <= self.max_chase_price:
                new_price = target_price
            else:
                new_price = self.max_chase_price
                
        else: # Sell
            # Target: Best Ask
            target_price = best_ask - self.chase_offset
            
            # Don't go below min price (which is stored in max_chase_price for simplicity logic here)
            if target_price >= self.max_chase_price:
                new_price = target_price
            else:
                new_price = self.max_chase_price
        
        # If price changed significantly, update
        if abs(new_price - self.price) > 0.0000001:
            logger.info(f"SmartOrder {self.order_id}: Adjusting price {self.price} -> {new_price}")
            self.price = new_price
            # In real system, this would trigger a replace_order call

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
    volume_profile: Optional[List[float]] = None  # List of volume percentages for each chunk

    def __post_init__(self):
        super().__post_init__()
        self.order_type = OrderType.VWAP
