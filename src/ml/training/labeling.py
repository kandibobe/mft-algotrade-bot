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

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TripleBarrierConfig:
    """Configuration for Triple Barrier labeling."""

    # Take profit threshold (e.g., 0.015 = 1.5%)
    take_profit: float = 0.015

    # Stop loss threshold (e.g., 0.0075 = 0.75%)
    stop_loss: float = 0.0075

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
            df: DataFrame with OHLCV data (MUST be time-sorted)

        Returns:
            Series with labels: 1 (TP hit), -1 (SL hit), 0 (time barrier/no clear signal)

        Raises:
            ValueError: If data validation fails (e.g., not time-sorted)
        """
        # CRITICAL: Validate data to prevent future data leakage
        self._validate_data(df)

        logger.info(f"Applying Triple Barrier labeling to {len(df)} rows")
        logger.info(
            f"Config: TP={self.config.take_profit:.3%}, "
            f"SL={self.config.stop_loss:.3%}, "
            f"Max hold={self.config.max_holding_period} bars"
        )

        labels = pd.Series(index=df.index, dtype=float)

        # Adjust barriers for fees
        tp_adjusted = self.config.take_profit - self.config.fee_adjustment
        sl_adjusted = self.config.stop_loss + self.config.fee_adjustment

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        for i in range(len(df) - self.config.max_holding_period):
            entry_price = close[i]

            # Upper and lower barriers
            upper_barrier = entry_price * (1 + tp_adjusted)
            lower_barrier = entry_price * (1 - sl_adjusted)

            # Scan forward to find which barrier is hit first
            label = self._get_barrier_label(
                high[i + 1 : i + 1 + self.config.max_holding_period],
                low[i + 1 : i + 1 + self.config.max_holding_period],
                close[i + 1 : i + 1 + self.config.max_holding_period],
                entry_price,
                upper_barrier,
                lower_barrier,
            )

            labels.iloc[i] = label

        # Fill remaining labels with NaN (not enough forward data)
        labels.iloc[-self.config.max_holding_period :] = np.nan

        # Log distribution
        label_counts = labels.value_counts()
        logger.info(f"Label distribution: {label_counts.to_dict()}")

        return labels

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate data to prevent future data leakage.

        Checks:
        - Data is sorted chronologically
        - No duplicate timestamps
        - Required OHLCV columns present
        - No NaN values in price data

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If validation fails
        """
        # Check required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. " f"DataFrame must have OHLCV data."
            )

        # CRITICAL: Check for time-based index or timestamp column
        has_datetime_index = isinstance(df.index, pd.DatetimeIndex)
        has_timestamp_col = "timestamp" in df.columns
        
        if not has_datetime_index and not has_timestamp_col:
            raise ValueError(
                "Data must have either a DatetimeIndex or a 'timestamp' column "
                "to ensure chronological ordering and prevent future data leakage. "
                "The labeling algorithm looks forward in time, so data MUST be "
                "sorted chronologically.\n"
                "Solutions:\n"
                "1. Set datetime as index: df.set_index('timestamp', inplace=True)\n"
                "2. Or ensure you have a 'timestamp' column with datetime values"
            )
        
        # Check chronological ordering
        if has_datetime_index:
            # CRITICAL: Check data is sorted by time (no future data leakage)
            if not df.index.is_monotonic_increasing:
                raise ValueError(
                    "Data is NOT sorted chronologically! This would cause future "
                    "data leakage where labels use information from the past. "
                    "Sort your data by timestamp before labeling:\n"
                    "df = df.sort_index()  # or df.sort_values('timestamp')"
                )

            # Check for duplicate timestamps
            if df.index.duplicated().any():
                dup_count = df.index.duplicated().sum()
                logger.warning(
                    f"Found {dup_count} duplicate timestamps. "
                    f"This may cause unexpected behavior in labeling."
                )
        elif has_timestamp_col:
            # Check if timestamp column is datetime type
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                logger.warning(
                    "Timestamp column is not datetime type. Converting to datetime."
                )
                try:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                except Exception as e:
                    raise ValueError(
                        f"Failed to convert 'timestamp' column to datetime: {e}"
                    )
            
            # Check chronological ordering using timestamp column
            if not df["timestamp"].is_monotonic_increasing:
                raise ValueError(
                    "Data is NOT sorted chronologically by 'timestamp' column! "
                    "This would cause future data leakage. Sort your data:\n"
                    "df = df.sort_values('timestamp')"
                )
            
            # Check for duplicate timestamps
            if df["timestamp"].duplicated().any():
                dup_count = df["timestamp"].duplicated().sum()
                logger.warning(
                    f"Found {dup_count} duplicate timestamps. "
                    f"This may cause unexpected behavior in labeling."
                )

        # Check for NaN in price data
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                raise ValueError(
                    f"Found {nan_count} NaN values in '{col}' column. "
                    f"Price data must not contain NaN values. "
                    f"Clean your data first:\n"
                    f"df = df.dropna(subset=['open', 'high', 'low', 'close'])"
                )

        # Check for invalid price relationships
        invalid_ohlc = (
            (df["high"] < df["low"])
            | (df["close"] > df["high"])
            | (df["close"] < df["low"])
            | (df["open"] > df["high"])
            | (df["open"] < df["low"])
        )

        if invalid_ohlc.any():
            invalid_count = invalid_ohlc.sum()
            logger.warning(
                f"Found {invalid_count} rows with invalid OHLC relationships "
                f"(e.g., high < low, close > high). Check your data quality."
            )

        logger.debug("✅ Data validation passed - safe for labeling")

    def _get_barrier_label(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        entry_price: float,
        upper_barrier: float,
        lower_barrier: float,
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
            # Check if BOTH barriers hit in same candle
            upper_hit = highs[j] >= upper_barrier
            lower_hit = lows[j] <= lower_barrier

            if upper_hit and lower_hit:
                # Both barriers hit - use close to determine which was "first"
                # This is a simplification; in reality would need tick data
                # If close > entry, assume TP was hit (price went up then down)
                # If close < entry, assume SL was hit (price went down then up)
                if closes[j] >= entry_price:
                    return 1
                else:
                    return -1

            # Only upper barrier hit (TP)
            if upper_hit:
                return 1

            # Only lower barrier hit (SL)
            if lower_hit:
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

    def _get_barrier_label_binary(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        entry_price: float,
        upper_barrier: float,
        lower_barrier: float,
    ) -> int:
        """
        Determine which barrier is hit first for BINARY classification.
        
        Binary labeling: 
        - Label = 1 (BUY) ONLY if price hits Take Profit BEFORE Stop Loss within N candles
        - Label = 0 (IGNORE) otherwise (SL hit first, time barrier, or both hit same candle)
        
        This ensures model only learns to predict truly profitable trades.

        Args:
            highs: High prices for forward period
            lows: Low prices for forward period
            closes: Close prices for forward period
            entry_price: Entry price
            upper_barrier: Take profit level
            lower_barrier: Stop loss level

        Returns:
            Label: 1 (TP hit before SL), 0 (otherwise)
        """
        for j in range(len(highs)):
            # Check if BOTH barriers hit in same candle
            upper_hit = highs[j] >= upper_barrier
            lower_hit = lows[j] <= lower_barrier

            if upper_hit and lower_hit:
                # Both barriers hit in same candle - ambiguous, treat as 0 (ignore)
                return 0

            # Only upper barrier hit (TP) - BUY signal
            if upper_hit:
                return 1

            # Only lower barrier hit (SL) - IGNORE
            if lower_hit:
                return 0

        # Time barrier hit - no clear signal, IGNORE
        return 0

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

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        for i in range(len(df) - self.config.max_holding_period):
            entry_price = close[i]
            upper_barrier = entry_price * (1 + tp_adjusted)
            lower_barrier = entry_price * (1 - sl_adjusted)

            label, barrier_type, holding_period, return_pct = self._get_barrier_details(
                high[i + 1 : i + 1 + self.config.max_holding_period],
                low[i + 1 : i + 1 + self.config.max_holding_period],
                close[i + 1 : i + 1 + self.config.max_holding_period],
                entry_price,
                upper_barrier,
                lower_barrier,
            )

            results.append(
                {
                    "label": label,
                    "barrier_type": barrier_type,
                    "holding_period": holding_period,
                    "return_pct": return_pct,
                }
            )

        # Add NaN rows for the end
        for _ in range(self.config.max_holding_period):
            results.append(
                {
                    "label": np.nan,
                    "barrier_type": "insufficient_data",
                    "holding_period": np.nan,
                    "return_pct": np.nan,
                }
            )

        result_df = pd.DataFrame(results, index=df.index)

        # Log statistics
        logger.info(
            f"Barrier type distribution: " f"{result_df['barrier_type'].value_counts().to_dict()}"
        )
        logger.info(f"Average holding period: " f"{result_df['holding_period'].mean():.1f} bars")

        return result_df

    def _get_barrier_details(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        entry_price: float,
        upper_barrier: float,
        lower_barrier: float,
    ) -> Tuple[int, str, int, float]:
        """Get detailed barrier information."""
        for j in range(len(highs)):
            # Check both barriers
            upper_hit = highs[j] >= upper_barrier
            lower_hit = lows[j] <= lower_barrier

            if upper_hit and lower_hit:
                # Both hit - use close to determine
                if closes[j] >= entry_price:
                    return_pct = (upper_barrier - entry_price) / entry_price
                    return 1, "take_profit", j + 1, return_pct
                else:
                    return_pct = (lower_barrier - entry_price) / entry_price
                    return -1, "stop_loss", j + 1, return_pct

            if upper_hit:
                return_pct = (upper_barrier - entry_price) / entry_price
                return 1, "take_profit", j + 1, return_pct

            if lower_hit:
                return_pct = (lower_barrier - entry_price) / entry_price
                return -1, "stop_loss", j + 1, return_pct

        # Time barrier
        return_pct = (closes[-1] - entry_price) / entry_price
        return 0, "time_barrier", len(closes), return_pct


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
        atr_multiplier_sl: float = 1.0,
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
        """Apply dynamic barrier labeling with BINARY classification."""
        logger.info("Applying Dynamic Triple Barrier labeling (binary)")

        # Calculate ATR
        atr = self._calculate_atr(df)
        atr_pct = atr / df["close"]  # ATR as percentage of price

        labels = pd.Series(index=df.index, dtype=float)

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

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

            label = self._get_barrier_label_binary(
                high[i + 1 : i + 1 + self.config.max_holding_period],
                low[i + 1 : i + 1 + self.config.max_holding_period],
                close[i + 1 : i + 1 + self.config.max_holding_period],
                entry_price,
                upper_barrier,
                lower_barrier,
            )

            labels.iloc[i] = label

        # Fill edges with NaN
        labels.iloc[: self.atr_period] = np.nan
        labels.iloc[-self.config.max_holding_period :] = np.nan

        label_counts = labels.value_counts()
        logger.info(f"Dynamic label distribution: {label_counts.to_dict()}")

        return labels

    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        return true_range.rolling(self.atr_period).mean()


def create_labels_for_training(
    df: pd.DataFrame, method: str = "triple_barrier", **kwargs
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
    if method == "simple":
        # Old simple method (not recommended)
        logger.warning("Using simple labeling - not recommended for production")
        return (df["close"].shift(-1) > df["close"]).astype(int)

    elif method == "triple_barrier":
        config = TripleBarrierConfig(**kwargs)
        labeler = TripleBarrierLabeler(config)
        return labeler.label(df)

    elif method == "dynamic_barrier":
        config = TripleBarrierConfig(
            take_profit=kwargs.pop("take_profit", 0.005),
            stop_loss=kwargs.pop("stop_loss", 0.002),
            max_holding_period=kwargs.pop("max_holding_period", 24),
            fee_adjustment=kwargs.pop("fee_adjustment", 0.001),
        )
        labeler = DynamicBarrierLabeler(config, **kwargs)
        return labeler.label(df)

    else:
        raise ValueError(f"Unknown labeling method: {method}")
