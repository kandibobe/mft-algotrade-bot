"""
Stoic Citadel - Market Regime Detection
========================================

Detect market regimes (trending/ranging, high/low volatility)
to adapt strategy behavior.

"The wise trader adapts to market conditions."
"""

import logging
from enum import Enum
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types."""

    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


def detect_trend_regime(
    close: pd.Series,
    ema_short: int = 50,
    ema_long: int = 200,
    adx_threshold: float = 25.0,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Detect trend regime using EMA crossover and ADX.

    Args:
        close: Close price series
        ema_short: Short EMA period
        ema_long: Long EMA period
        adx_threshold: ADX threshold for trending market
        high: High price series (for ADX)
        low: Low price series (for ADX)

    Returns:
        Series with regime labels
    """
    from .indicators import calculate_adx, calculate_ema

    # Calculate EMAs
    ema_s = calculate_ema(close, ema_short)
    ema_l = calculate_ema(close, ema_long)

    # Calculate ADX if high/low provided
    if high is not None and low is not None:
        adx_data = calculate_adx(high, low, close)
        adx = adx_data["adx"]
        is_trending = adx > adx_threshold
    else:
        # Fallback: use EMA slope as trend indicator
        ema_slope = ema_l.diff(5) / ema_l * 100
        is_trending = ema_slope.abs() > 0.1

    # Determine regime
    regime = pd.Series(index=close.index, dtype=str)

    bull_trend = (ema_s > ema_l) & is_trending
    bear_trend = (ema_s < ema_l) & is_trending
    ranging = ~is_trending

    regime[bull_trend] = MarketRegime.TRENDING_BULL.value
    regime[bear_trend] = MarketRegime.TRENDING_BEAR.value
    regime[ranging] = MarketRegime.RANGING.value

    return regime


def detect_volatility_regime(
    close: pd.Series,
    lookback: int = 30,
    high_vol_percentile: float = 75,
    low_vol_percentile: float = 25,
) -> pd.Series:
    """
    Detect volatility regime using realized volatility.

    Args:
        close: Close price series
        lookback: Period for volatility calculation
        high_vol_percentile: Percentile threshold for high vol
        low_vol_percentile: Percentile threshold for low vol

    Returns:
        Series with volatility regime labels
    """
    # Calculate rolling realized volatility
    returns = close.pct_change()
    realized_vol = returns.rolling(window=lookback).std() * np.sqrt(252)

    # Calculate rolling percentiles
    vol_rank = realized_vol.rolling(window=lookback * 5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )

    # Classify
    regime = pd.Series(index=close.index, dtype=str)

    high_vol = vol_rank > high_vol_percentile
    low_vol = vol_rank < low_vol_percentile
    normal_vol = ~high_vol & ~low_vol

    regime[high_vol] = MarketRegime.HIGH_VOLATILITY.value
    regime[low_vol] = MarketRegime.LOW_VOLATILITY.value
    regime[normal_vol] = "normal_volatility"

    return regime


def calculate_regime_score(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.DataFrame:
    """
    Calculate comprehensive regime scores.

    Returns DataFrame with multiple regime indicators.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        volume: Volume series

    Returns:
        DataFrame with regime scores and indicators
    """
    from .indicators import calculate_adx, calculate_atr, calculate_rsi

    result = pd.DataFrame(index=close.index)

    # Trend score (0-100, >50 = bullish)
    ema_50 = close.ewm(span=50).mean()
    ema_200 = close.ewm(span=200).mean()

    result["ema_trend"] = (ema_50 > ema_200).astype(int)
    result["price_vs_ema"] = (close > ema_50).astype(int)

    # ADX for trend strength
    adx_data = calculate_adx(high, low, close)
    result["adx"] = adx_data["adx"]
    result["trend_strength"] = result["adx"].clip(0, 100) / 100

    # Volatility score
    atr = calculate_atr(high, low, close)
    result["atr_pct"] = atr / close * 100

    # Rolling volatility percentile
    vol_lookback = 100
    result["volatility_rank"] = (
        result["atr_pct"]
        .rolling(vol_lookback)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    )

    # RSI for overbought/oversold
    result["rsi"] = calculate_rsi(close)

    # Volume analysis
    result["volume_ratio"] = volume / volume.rolling(20).mean()

    # Composite regime score
    # Bullish: 0-100 scale
    # >60 = strongly bullish, 40-60 = neutral, <40 = bearish
    result["regime_score"] = (
        result["ema_trend"] * 30
        + result["price_vs_ema"] * 20
        + (result["rsi"] > 50).astype(int) * 20
        + (result["adx"] > 25).astype(int) * 15
        + (result["volume_ratio"] > 1).astype(int) * 15
    )

    # Risk adjustment factor (1.0 = normal, <1 = reduce risk)
    result["risk_factor"] = np.where(
        result["volatility_rank"] > 0.8,
        0.5,  # High volatility: reduce risk
        np.where(
            result["volatility_rank"] < 0.2, 1.2, 1.0  # Low volatility: can increase risk  # Normal
        ),
    )

    return result


def get_regime_parameters(
    regime_score: float, base_risk: float = 0.02, base_leverage: float = 1.0
) -> Dict[str, float]:
    """
    Get recommended trading parameters based on regime.

    Args:
        regime_score: Composite regime score (0-100)
        base_risk: Base risk per trade
        base_leverage: Base leverage

    Returns:
        Dict with adjusted parameters
    """
    if regime_score > 70:
        # Strongly bullish: aggressive
        return {
            "risk_per_trade": base_risk * 1.2,
            "leverage": min(base_leverage * 1.5, 3.0),
            "min_rsi_entry": 25,  # Buy deeper dips
            "max_positions": 5,
            "mode": "aggressive",
        }
    elif regime_score > 55:
        # Moderately bullish: normal
        return {
            "risk_per_trade": base_risk,
            "leverage": base_leverage,
            "min_rsi_entry": 30,
            "max_positions": 4,
            "mode": "normal",
        }
    elif regime_score > 40:
        # Neutral: cautious
        return {
            "risk_per_trade": base_risk * 0.75,
            "leverage": base_leverage * 0.8,
            "min_rsi_entry": 25,
            "max_positions": 3,
            "mode": "cautious",
        }
    else:
        # Bearish: defensive
        return {
            "risk_per_trade": base_risk * 0.5,
            "leverage": base_leverage * 0.5,
            "min_rsi_entry": 20,
            "max_positions": 2,
            "mode": "defensive",
        }
