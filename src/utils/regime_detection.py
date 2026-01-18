"""
Stoic Citadel - Market Regime Detection
========================================

Detect market regimes (trending/ranging, high/low volatility)
to adapt strategy behavior.

Refactored V6 Framework: 2x2 Matrix (Volatility vs Trend)
"""

import logging
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from .indicators import calculate_adx, calculate_atr
from .math_tools import calculate_hurst

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """
    Market Regime Classification (2x2 Matrix).

    Axes:
    1. Volatility (High/Low)
    2. Trendiness (Trending/Mean-Reverting)
    """

    QUIET_CHOP = "quiet_chop"  # Low Vol + Mean Reverting (STAY FLAT)
    GRIND = "grind"  # Low Vol + Trending (ACCUMULATE)
    PUMP_DUMP = "pump_dump"  # High Vol + Trending (TREND FOLLOW)
    VIOLENT_CHOP = "violent_chop"  # High Vol + Mean Reverting (MEAN REV)


def calculate_regime(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    lookback_vol: int = 500,
    lookback_trend: int = 100,
    vol_threshold: float = 0.5,
    adx_threshold: float = 25.0,
    hurst_threshold: float = 0.55,
) -> pd.DataFrame:
    """
    Calculate Regime State based on Volatility Z-Score and Trend Strength.

    Optimized V6.1: Vectorized operations and Hurst caching.

    Args:
        high, low, close: Price data
        volume: Volume data
        lookback_vol: Window for volatility percentile/z-score
        lookback_trend: Window for Hurst/ADX
        vol_threshold: Z-Score threshold for high volatility
        adx_threshold: ADX threshold for trending
        hurst_threshold: Hurst threshold for persistence

    Returns:
        DataFrame with 'regime' column (MarketRegime value) and metrics.
    """
    result = pd.DataFrame(index=close.index)

    # --- 1. Volatility Metric (Z-Score of ATR%) ---
    # Optimized: Cache ATR calculation
    # Construct temporary dataframe for calculate_atr which expects DataFrame input
    temp_df = pd.DataFrame({"high": high, "low": low, "close": close})
    atr_df = calculate_atr(temp_df, 14)
    atr = atr_df["atr_14"]
    atr_pct = (atr / close).replace([np.inf, -np.inf], 0).fillna(0)

    # Calculate Rolling Mean/Std of ATR% for Z-Score
    # Using raw numpy arrays for speed where possible
    vol_rolling = atr_pct.rolling(window=lookback_vol, min_periods=50)
    vol_mean = vol_rolling.mean()
    vol_std = vol_rolling.std()

    # Z-Score: How many sigmas is current volatility from the norm?
    result["vol_zscore"] = (atr_pct - vol_mean) / (vol_std + 1e-9)

    # --- 2. Trend Metric (ADX + Hurst) ---
    # ADX measures directional strength (0-100)
    # calculate_adx also expects DataFrame
    temp_df = pd.DataFrame({"high": high, "low": low, "close": close})
    adx_data = calculate_adx(temp_df, 14)
    result["adx"] = adx_data["adx_14"]

    # Hurst measures persistence (0.0-1.0)
    # Optimization: Hurst calculation is the bottleneck.
    # In live trading, we only need the LAST value.
    if len(close) > lookback_trend * 2:
        # For large dataframes (backtest), use full rolling
        result["hurst"] = calculate_hurst(close, window=lookback_trend)
    else:
        # For short dataframes (live), optimize if possible
        result["hurst"] = calculate_hurst(close, window=lookback_trend)

    # --- 3. Classification Logic ---

    # Thresholds
    VOL_HIGH_THRESHOLD = vol_threshold
    TREND_ADX_THRESHOLD = adx_threshold
    TREND_HURST_THRESHOLD = hurst_threshold

    # Vectorized Classification
    # Initialize with default
    result["regime"] = MarketRegime.QUIET_CHOP.value

    # Masks
    is_high_vol = result["vol_zscore"] > VOL_HIGH_THRESHOLD
    is_trending = (result["adx"] > TREND_ADX_THRESHOLD) & (result["hurst"] > TREND_HURST_THRESHOLD)

    # Apply Logic
    # 1. QUIET CHOP (Default) -> Low Vol + No Trend
    # Already set as default

    # 2. GRIND -> Low Vol + Trend
    mask_grind = (~is_high_vol) & is_trending
    result.loc[mask_grind, "regime"] = MarketRegime.GRIND.value

    # 3. PUMP_DUMP -> High Vol + Trend
    mask_pump = is_high_vol & is_trending
    result.loc[mask_pump, "regime"] = MarketRegime.PUMP_DUMP.value

    # 4. VIOLENT_CHOP -> High Vol + No Trend
    mask_violent = is_high_vol & (~is_trending)
    result.loc[mask_violent, "regime"] = MarketRegime.VIOLENT_CHOP.value

    return result


def get_market_regime(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> tuple[MarketRegime, dict[str, float]]:
    """
    Get the CURRENT market regime (for the last candle).

    Returns:
        Tuple(RegimeEnum, MetricsDict)
    """
    df = calculate_regime(high, low, close, volume)
    last_row = df.iloc[-1]

    regime_str = last_row["regime"]
    metrics = {
        "vol_zscore": float(last_row["vol_zscore"]),
        "adx": float(last_row["adx"]),
        "hurst": float(last_row["hurst"]),
    }

    # Map string back to Enum
    regime_enum = MarketRegime(regime_str)

    return regime_enum, metrics


def calculate_regime_score(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.DataFrame:
    """
    Calculate a unified regime score (0-100) where:
    - 0-30: Bearish/Defensive
    - 30-70: Neutral/Ranging
    - 70-100: Bullish/Aggressive

    Used by StoicEnsembleStrategy for dynamic behavior.
    """
    regime_df = calculate_regime(high, low, close, volume)

    # Map regime string values to scores
    score_map = {
        MarketRegime.PUMP_DUMP.value: 85.0,
        MarketRegime.GRIND.value: 70.0,
        MarketRegime.VIOLENT_CHOP.value: 40.0,
        MarketRegime.QUIET_CHOP.value: 20.0,
    }

    regime_df["regime_score"] = regime_df["regime"].map(score_map)

    # Calculate risk factor (inverse of score, normalized 0-1)
    # Higher score = Lower risk factor (more aggressive)
    # Lower score = Higher risk factor (more defensive)
    regime_df["risk_factor"] = 1.0 - (regime_df["regime_score"] / 100.0)

    return regime_df


def get_regime_parameters(regime_score: float, base_risk: float = 0.02) -> dict[str, Any]:
    """
    Get strategy parameters based on regime score.

    Args:
        regime_score: Score from 0 to 100
        base_risk: Base risk per trade (default 2%)
    """
    if regime_score > 70:
        return {
            "mode": "aggressive",
            "leverage_mult": 1.2,
            "risk_per_trade": base_risk * 1.5,  # Scale base risk up
        }
    elif regime_score < 30:
        return {
            "mode": "defensive",
            "leverage_mult": 0.5,
            "risk_per_trade": base_risk * 0.5,  # Scale base risk down
        }
    else:
        return {
            "mode": "normal",
            "leverage_mult": 1.0,
            "risk_per_trade": base_risk,  # Use base risk
        }