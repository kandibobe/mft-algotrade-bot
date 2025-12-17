"""
Stoic Citadel - Slippage Model
==============================

Realistic slippage simulation for backtesting:
- Volume-based slippage
- Volatility-based slippage
- Time-of-day effects
- Market impact modeling
"""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SlippageConfig:
    """Slippage model configuration."""
    # Fixed slippage
    fixed_slippage_pct: float = 0.0005  # 0.05% fixed
    
    # Volume-based
    volume_impact_factor: float = 0.1   # Impact per % of volume
    max_volume_pct: float = 0.05        # Max order as % of volume
    
    # Volatility-based  
    volatility_multiplier: float = 0.5  # Scale slippage with volatility
    
    # Spread
    avg_spread_pct: float = 0.001       # Average bid-ask spread
    
    # Time-of-day multipliers
    market_open_multiplier: float = 2.0  # Higher slippage at open
    market_close_multiplier: float = 1.5 # Higher slippage at close
    
    # Random component
    random_factor: float = 0.2          # Random variation percentage


class SlippageModel:
    """
    Realistic slippage model for backtesting.
    
    Combines multiple factors:
    1. Fixed component (minimum slippage)
    2. Volume-based impact (larger orders have more slippage)
    3. Volatility adjustment (high vol = more slippage)
    4. Time-of-day effects
    5. Random component (market microstructure noise)
    """
    
    def __init__(self, config: Optional[SlippageConfig] = None):
        self.config = config or SlippageConfig()
    
    def calculate_slippage(
        self,
        order_size: float,
        price: float,
        side: str,  # "buy" or "sell"
        volume: Optional[float] = None,
        volatility: Optional[float] = None,
        spread: Optional[float] = None,
        hour_of_day: Optional[int] = None
    ) -> float:
        """
        Calculate estimated slippage for an order.
        
        Args:
            order_size: Order quantity
            price: Current price
            side: "buy" or "sell"
            volume: Average volume (optional)
            volatility: Recent volatility (optional)
            spread: Current bid-ask spread (optional)
            hour_of_day: Hour of day 0-23 (optional)
            
        Returns:
            Slippage in price units (always positive)
        """
        order_value = order_size * price
        
        # 1. Fixed component
        fixed_slippage = price * self.config.fixed_slippage_pct
        
        # 2. Spread component (half spread)
        spread_cost = spread or (price * self.config.avg_spread_pct)
        spread_slippage = spread_cost / 2
        
        # 3. Volume-based impact
        volume_slippage = 0.0
        if volume and volume > 0:
            volume_ratio = (order_size / volume)
            if volume_ratio > self.config.max_volume_pct:
                # Warn about large orders
                logger.warning(
                    f"Large order: {volume_ratio:.1%} of volume"
                )
            volume_slippage = price * volume_ratio * self.config.volume_impact_factor
        
        # 4. Volatility adjustment
        vol_multiplier = 1.0
        if volatility:
            # Higher volatility = more slippage
            vol_multiplier = 1 + (volatility * self.config.volatility_multiplier)
        
        # 5. Time-of-day adjustment
        time_multiplier = 1.0
        if hour_of_day is not None:
            if hour_of_day < 10:  # First hour of trading
                time_multiplier = self.config.market_open_multiplier
            elif hour_of_day >= 15:  # Last hour
                time_multiplier = self.config.market_close_multiplier
        
        # 6. Random component
        random_factor = 1 + np.random.uniform(
            -self.config.random_factor,
            self.config.random_factor
        )
        
        # Combine all components
        total_slippage = (
            (fixed_slippage + spread_slippage + volume_slippage) *
            vol_multiplier *
            time_multiplier *
            random_factor
        )
        
        return max(0, total_slippage)
    
    def calculate_market_impact(
        self,
        order_value: float,
        market_cap: Optional[float] = None,
        daily_volume: Optional[float] = None
    ) -> float:
        """
        Calculate market impact of order (for very large orders).
        
        Uses square-root market impact model:
        Impact = c * sqrt(Q / V)
        
        Where:
            c = impact coefficient
            Q = order value
            V = daily volume
        """
        if not daily_volume or daily_volume <= 0:
            return 0.0
        
        c = 0.1  # Impact coefficient
        impact = c * np.sqrt(order_value / daily_volume)
        
        return impact
    
    def estimate_fill_price(
        self,
        current_price: float,
        order_size: float,
        side: str,
        **kwargs
    ) -> float:
        """
        Estimate fill price including slippage.
        
        Args:
            current_price: Current market price
            order_size: Order quantity
            side: "buy" or "sell"
            **kwargs: Additional parameters for slippage calculation
            
        Returns:
            Estimated fill price
        """
        slippage = self.calculate_slippage(
            order_size=order_size,
            price=current_price,
            side=side,
            **kwargs
        )
        
        if side.lower() == "buy":
            return current_price + slippage
        else:
            return current_price - slippage
    
    def apply_slippage_to_series(
        self,
        df: pd.DataFrame,
        entry_signals: pd.Series,
        exit_signals: pd.Series,
        position_size: float
    ) -> pd.DataFrame:
        """
        Apply slippage to backtest entry/exit signals.
        
        Args:
            df: DataFrame with OHLCV data
            entry_signals: Boolean series for entries
            exit_signals: Boolean series for exits
            position_size: Fixed position size
            
        Returns:
            DataFrame with adjusted prices
        """
        result = df.copy()
        result['entry_slippage'] = 0.0
        result['exit_slippage'] = 0.0
        result['entry_fill_price'] = result['close']
        result['exit_fill_price'] = result['close']
        
        for i in range(len(result)):
            # Calculate volatility from recent data
            if i >= 20:
                returns = result['close'].iloc[i-20:i].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)
            else:
                volatility = 0.02  # Default
            
            volume = result['volume'].iloc[i] if 'volume' in result.columns else None
            hour = result.index[i].hour if hasattr(result.index[i], 'hour') else None
            
            if entry_signals.iloc[i]:
                slippage = self.calculate_slippage(
                    order_size=position_size,
                    price=result['close'].iloc[i],
                    side="buy",
                    volume=volume,
                    volatility=volatility,
                    hour_of_day=hour
                )
                result.loc[result.index[i], 'entry_slippage'] = slippage
                result.loc[result.index[i], 'entry_fill_price'] = (
                    result['close'].iloc[i] + slippage
                )
            
            if exit_signals.iloc[i]:
                slippage = self.calculate_slippage(
                    order_size=position_size,
                    price=result['close'].iloc[i],
                    side="sell",
                    volume=volume,
                    volatility=volatility,
                    hour_of_day=hour
                )
                result.loc[result.index[i], 'exit_slippage'] = slippage
                result.loc[result.index[i], 'exit_fill_price'] = (
                    result['close'].iloc[i] - slippage
                )
        
        return result
    
    def estimate_execution_cost(
        self,
        order_value: float,
        commission_rate: float = 0.001,
        **kwargs
    ) -> dict:
        """
        Estimate total execution cost including commission.
        
        Returns breakdown of costs.
        """
        # Get slippage using an average price assumption
        price = 100  # Normalized price
        quantity = order_value / price
        
        slippage = self.calculate_slippage(
            order_size=quantity,
            price=price,
            side="buy",
            **kwargs
        )
        
        slippage_cost = slippage * quantity
        commission = order_value * commission_rate
        
        return {
            "order_value": order_value,
            "slippage_pct": (slippage / price) * 100,
            "slippage_cost": slippage_cost,
            "commission_pct": commission_rate * 100,
            "commission": commission,
            "total_cost": slippage_cost + commission,
            "total_cost_pct": ((slippage_cost + commission) / order_value) * 100
        }
