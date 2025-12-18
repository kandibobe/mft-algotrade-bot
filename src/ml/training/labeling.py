"""
Triple Barrier Labeling
========================

Advanced labeling method for trading ML models.
Uses take-profit, stop-loss, and time barriers to create meaningful labels.

This solves the fundamental problem of naive labeling where:
- Simple "next candle up/down" labels ignore transaction costs
- Model can be "right" but still lose money due to fees

Reference: Advances in Financial Machine Learning, Marcos Lopez de Prado
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class TripleBarrierConfig:
    """Configuration for Triple Barrier labeling."""

    # Take profit threshold (e.g., 0.005 = 0.5%)
    take_profit: float = 0.005

    # Stop loss threshold (e.g., 0.002 = 0.2%)
    stop_loss: float = 0.002

    # Maximum bars to hold position
    max_holding_period: int = 24  # 24 bars = 2 hours for 5m timeframe

    # Minimum price movement to consider (filters noise)
    min_movement: float = 0.001

    # Include "hold" class (no significant movement)
    include_hold_class: bool = True

    # Account for trading fees in barriers
    fee_adjustment: float = 0.001  # 0.1% round-trip fees


class TripleBarrierLabeler:
    """
    Triple Barrier Method for labeling trading data.

    Creates labels based on which barrier is hit first:
    - Upper barrier (take profit): Label = 1 (Buy signal)
    - Lower barrier (stop loss): Label = -1 (Sell/Avoid signal)
    - Time barrier (max holding): Label = 0 (Hold/No signal)

    This method ensures that:
    1. Positive labels only when trade is actually profitable after fees
    2. Model learns to avoid unprofitable situations
    3. Hold signals for low-confidence periods

    Usage:
        labeler = TripleBarrierLabeler(config)
        labels = labeler.label(df)
    """

    def __init__(self, config: Optional[TripleBarrierConfig] = None):
        """Initialize labeler with config."""
        self.config = config or TripleBarrierConfig()

    def label(self, df: pd.DataFrame) -> pd.Series:
        """
        Apply Triple Barrier Method to create labels.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series with labels: 1 (buy), 0 (hold), -1 (sell/avoid)
        """
        logger.info(f"Applying Triple Barrier labeling to {len(df)} rows")
        logger.info(f"Config: TP={self.config.take_profit:.3%}, "
                   f"SL={self.config.stop_loss:.3%}, "
                   f"Max hold={self.config.max_holding_period} bars")

        labels = pd.Series(index=df.index, dtype=float)

        # Adjust barriers for fees
        tp_adjusted = self.config.take_profit - self.config.fee_adjustment
        sl_adjusted = self.config.stop_loss + self.config.fee_adjustment

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        for i in range(len(df) - self.config.max_holding_period):
            entry_price = close[i]

            # Upper and lower barriers
            upper_barrier = entry_price * (1 + tp_adjusted)
            lower_barrier = entry_price * (1 - sl_adjusted)

            # Scan forward to find which barrier is hit first
            label = self._get_barrier_label(
                high[i+1:i+1+self.config.max_holding_period],
                low[i+1:i+1+self.config.max_holding_period],
                close[i+1:i+1+self.config.max_holding_period],
                entry_price,
                upper_barrier,
                lower_barrier
            )

            labels.iloc[i] = label

        # Fill remaining labels with NaN (not enough forward data)
        labels.iloc[-self.config.max_holding_period:] = np.nan

        # Log distribution
        label_counts = labels.value_counts()
        logger.info(f"Label distribution: {label_counts.to_dict()}")

        return labels

    def _get_barrier_label(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        entry_price: float,
        upper_barrier: float,
        lower_barrier: float
    ) -> int:
        """
        Determine which barrier is hit first.

        Args:
            highs: High prices for forward period
            lows: Low prices for forward period
            closes: Close prices for forward period
            entry_price: Entry price
            upper_barrier: Take profit level
            lower_barrier: Stop loss level

        Returns:
            Label: 1 (hit TP), -1 (hit SL), 0 (time barrier/no clear signal)
        """
        for j in range(len(highs)):
            # Check if upper barrier hit (TP)
            if highs[j] >= upper_barrier:
                # Also check if lower barrier hit in same bar
                if lows[j] <= lower_barrier:
                    # Both hit - use close to determine which was first
                    # This is a simplification; in reality would need tick data
                    if closes[j] >= entry_price:
                        return 1
                    else:
                        return -1
                return 1

            # Check if lower barrier hit (SL)
            if lows[j] <= lower_barrier:
                return -1

        # Time barrier hit - no clear signal
        if self.config.include_hold_class:
            # Check final movement direction for weak signal
            final_return = (closes[-1] - entry_price) / entry_price
            if abs(final_return) < self.config.min_movement:
                return 0  # No significant movement - hold
            elif final_return > 0:
                return 0  # Positive but didn't hit TP - weak signal
            else:
                return 0  # Negative but didn't hit SL - weak signal

        # Binary classification mode
        return 1 if closes[-1] > entry_price else -1

    def label_with_meta(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create labels with metadata for analysis.

        Returns DataFrame with:
        - label: The barrier label
        - barrier_type: Which barrier was hit
        - holding_period: How many bars until barrier
        - return_pct: Actual return achieved

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with labels and metadata
        """
        logger.info("Generating labels with metadata")

        results = []

        tp_adjusted = self.config.take_profit - self.config.fee_adjustment
        sl_adjusted = self.config.stop_loss + self.config.fee_adjustment

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        for i in range(len(df) - self.config.max_holding_period):
            entry_price = close[i]
            upper_barrier = entry_price * (1 + tp_adjusted)
            lower_barrier = entry_price * (1 - sl_adjusted)

            label, barrier_type, holding_period, return_pct = self._get_barrier_details(
                high[i+1:i+1+self.config.max_holding_period],
                low[i+1:i+1+self.config.max_holding_period],
                close[i+1:i+1+self.config.max_holding_period],
                entry_price,
                upper_barrier,
                lower_barrier
            )

            results.append({
                'label': label,
                'barrier_type': barrier_type,
                'holding_period': holding_period,
                'return_pct': return_pct
            })

        # Add NaN rows for the end
        for _ in range(self.config.max_holding_period):
            results.append({
                'label': np.nan,
                'barrier_type': 'insufficient_data',
                'holding_period': np.nan,
                'return_pct': np.nan
            })

        result_df = pd.DataFrame(results, index=df.index)

        # Log statistics
        logger.info(f"Barrier type distribution: "
                   f"{result_df['barrier_type'].value_counts().to_dict()}")
        logger.info(f"Average holding period: "
                   f"{result_df['holding_period'].mean():.1f} bars")

        return result_df

    def _get_barrier_details(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        entry_price: float,
        upper_barrier: float,
        lower_barrier: float
    ) -> Tuple[int, str, int, float]:
        """Get detailed barrier information."""
        for j in range(len(highs)):
            if highs[j] >= upper_barrier:
                return_pct = (upper_barrier - entry_price) / entry_price
                return 1, 'take_profit', j + 1, return_pct

            if lows[j] <= lower_barrier:
                return_pct = (lower_barrier - entry_price) / entry_price
                return -1, 'stop_loss', j + 1, return_pct

        # Time barrier
        return_pct = (closes[-1] - entry_price) / entry_price
        return 0, 'time_barrier', len(closes), return_pct


class DynamicBarrierLabeler(TripleBarrierLabeler):
    """
    Dynamic Triple Barrier that adjusts thresholds based on volatility.

    Uses ATR to scale barriers, so:
    - High volatility periods have wider barriers
    - Low volatility periods have tighter barriers

    This prevents:
    - Too many stop-outs in volatile markets
    - Too few signals in quiet markets
    """

    def __init__(
        self,
        config: Optional[TripleBarrierConfig] = None,
        atr_period: int = 14,
        atr_multiplier_tp: float = 2.0,
        atr_multiplier_sl: float = 1.0
    ):
        """
        Initialize dynamic labeler.

        Args:
            config: Base configuration
            atr_period: Period for ATR calculation
            atr_multiplier_tp: ATR multiplier for take profit
            atr_multiplier_sl: ATR multiplier for stop loss
        """
        super().__init__(config)
        self.atr_period = atr_period
        self.atr_multiplier_tp = atr_multiplier_tp
        self.atr_multiplier_sl = atr_multiplier_sl

    def label(self, df: pd.DataFrame) -> pd.Series:
        """Apply dynamic barrier labeling."""
        logger.info("Applying Dynamic Triple Barrier labeling")

        # Calculate ATR
        atr = self._calculate_atr(df)
        atr_pct = atr / df['close']  # ATR as percentage of price

        labels = pd.Series(index=df.index, dtype=float)

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        for i in range(self.atr_period, len(df) - self.config.max_holding_period):
            entry_price = close[i]

            # Dynamic barriers based on ATR
            current_atr_pct = atr_pct.iloc[i]
            tp = max(current_atr_pct * self.atr_multiplier_tp, self.config.take_profit)
            sl = max(current_atr_pct * self.atr_multiplier_sl, self.config.stop_loss)

            # Adjust for fees
            tp_adjusted = tp - self.config.fee_adjustment
            sl_adjusted = sl + self.config.fee_adjustment

            upper_barrier = entry_price * (1 + tp_adjusted)
            lower_barrier = entry_price * (1 - sl_adjusted)

            label = self._get_barrier_label(
                high[i+1:i+1+self.config.max_holding_period],
                low[i+1:i+1+self.config.max_holding_period],
                close[i+1:i+1+self.config.max_holding_period],
                entry_price,
                upper_barrier,
                lower_barrier
            )

            labels.iloc[i] = label

        # Fill edges with NaN
        labels.iloc[:self.atr_period] = np.nan
        labels.iloc[-self.config.max_holding_period:] = np.nan

        label_counts = labels.value_counts()
        logger.info(f"Dynamic label distribution: {label_counts.to_dict()}")

        return labels

    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        return true_range.rolling(self.atr_period).mean()


def create_labels_for_training(
    df: pd.DataFrame,
    method: str = 'triple_barrier',
    **kwargs
) -> pd.Series:
    """
    Factory function to create labels.

    Args:
        df: DataFrame with OHLCV data
        method: Labeling method ('triple_barrier', 'dynamic_barrier', 'simple')
        **kwargs: Additional arguments for the labeler

    Returns:
        Series with labels
    """
    if method == 'simple':
        # Old simple method (not recommended)
        logger.warning("Using simple labeling - not recommended for production")
        return (df['close'].shift(-1) > df['close']).astype(int)

    elif method == 'triple_barrier':
        config = TripleBarrierConfig(**kwargs)
        labeler = TripleBarrierLabeler(config)
        return labeler.label(df)

    elif method == 'dynamic_barrier':
        config = TripleBarrierConfig(
            take_profit=kwargs.pop('take_profit', 0.005),
            stop_loss=kwargs.pop('stop_loss', 0.002),
            max_holding_period=kwargs.pop('max_holding_period', 24),
            fee_adjustment=kwargs.pop('fee_adjustment', 0.001)
        )
        labeler = DynamicBarrierLabeler(config, **kwargs)
        return labeler.label(df)

    else:
        raise ValueError(f"Unknown labeling method: {method}")
