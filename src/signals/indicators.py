"""
Shared Indicator Library
=========================

**CRITICAL**: This module is used by BOTH:
1. Research environment (Jupyter + VectorBT)
2. Production trading (Freqtrade strategies)

DO NOT modify calculation logic without updating tests and backtests.

Author: Stoic Citadel Team
License: MIT
"""

from typing import Tuple

import numpy as np
import pandas as pd
import talib.abstract as ta


class IndicatorLibrary:
    """
    Centralized indicator calculations guaranteeing research/production parity.

    Design Philosophy:
    - Pure functions (no side effects)
    - Type hints on all inputs/outputs
    - Vectorized calculations (no loops)
    - Unit tested with fixtures
    """

    @staticmethod
    def calculate_ema_trio(
        close: pd.Series, fast: int = 50, medium: int = 100, slow: int = 200
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate EMA trio for trend detection.

        Args:
            close: Close prices
            fast: Fast EMA period (default: 50)
            medium: Medium EMA period (default: 100)
            slow: Slow EMA period (default: 200)

        Returns:
            Tuple of (ema_fast, ema_medium, ema_slow)
        """
        ema_fast = ta.EMA(close, timeperiod=fast)
        ema_medium = ta.EMA(close, timeperiod=medium)
        ema_slow = ta.EMA(close, timeperiod=slow)
        return ema_fast, ema_medium, ema_slow

    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        return ta.RSI(close, timeperiod=period)

    @staticmethod
    def calculate_adx(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Calculate ADX (trend strength)."""
        return ta.ADX(high=high, low=low, close=close, timeperiod=period)

    @staticmethod
    def calculate_stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        fastk_period: int = 14,
        slowk_period: int = 3,
        slowd_period: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.

        Returns:
            Tuple of (slowk, slowd)
        """
        stoch = ta.STOCH(
            high=high,
            low=low,
            close=close,
            fastk_period=fastk_period,
            slowk_period=slowk_period,
            slowd_period=slowd_period,
        )
        return stoch["slowk"], stoch["slowd"]

    @staticmethod
    def calculate_macd(
        close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD.

        Returns:
            Tuple of (macd, macdsignal, macdhist)
        """
        macd_result = ta.MACD(close, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        return (macd_result["macd"], macd_result["macdsignal"], macd_result["macdhist"])

    @staticmethod
    def calculate_bollinger_bands(
        close: pd.Series, period: int = 20, nbdevup: float = 2.0, nbdevdn: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Returns:
            Tuple of (upper, middle, lower)
        """
        bb = ta.BBANDS(close, timeperiod=period, nbdevup=nbdevup, nbdevdn=nbdevdn)
        return bb["upperband"], bb["middleband"], bb["lowerband"]

    @staticmethod
    def calculate_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Calculate Average True Range (volatility)."""
        return ta.ATR(high=high, low=low, close=close, timeperiod=period)

    @staticmethod
    def calculate_trend_score(
        close: pd.Series, ema_fast: pd.Series, ema_medium: pd.Series, ema_slow: pd.Series
    ) -> pd.Series:
        """
        Calculate trend strength score (0-3).

        Score components:
        - +1 if price > EMA_fast
        - +1 if EMA_fast > EMA_medium
        - +1 if EMA_medium > EMA_slow
        """
        score = pd.Series(0, index=close.index)
        score += (close > ema_fast).astype(int)
        score += (ema_fast > ema_medium).astype(int)
        score += (ema_medium > ema_slow).astype(int)
        return score


class SignalGenerator:
    """
    Generate trading signals using indicator library.

    This class MUST produce identical signals in both research and production.
    """

    def __init__(self):
        self.indicators = IndicatorLibrary()

    def populate_all_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators on dataframe.

        Args:
            dataframe: OHLCV dataframe

        Returns:
            Dataframe with indicator columns added
        """
        df = dataframe.copy()

        # Trend indicators
        df["ema_50"], df["ema_100"], df["ema_200"] = self.indicators.calculate_ema_trio(
            df["close"], 50, 100, 200
        )

        # Oscillators
        df["rsi"] = self.indicators.calculate_rsi(df["close"])
        df["slowk"], df["slowd"] = self.indicators.calculate_stochastic(
            df["high"], df["low"], df["close"]
        )

        # Trend strength
        df["adx"] = self.indicators.calculate_adx(df["high"], df["low"], df["close"])

        # MACD
        df["macd"], df["macdsignal"], df["macdhist"] = self.indicators.calculate_macd(df["close"])

        # Bollinger Bands
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = self.indicators.calculate_bollinger_bands(
            df["close"]
        )

        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        # Volatility
        df["atr"] = self.indicators.calculate_atr(df["high"], df["low"], df["close"])

        # Volume
        df["volume_mean"] = df["volume"].rolling(window=20).mean()

        # Custom features
        df["pct_change_1"] = df["close"].pct_change(1)
        df["pct_change_3"] = df["close"].pct_change(3)
        df["pct_change_5"] = df["close"].pct_change(5)

        df["trend_score"] = self.indicators.calculate_trend_score(
            df["close"], df["ema_50"], df["ema_100"], df["ema_200"]
        )

        return df

    def generate_entry_signal(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Generate entry signals (1 = enter, 0 = no action).

        Logic:
        1. Trend filter: price > EMA200, EMA50 > EMA100, ADX > 20
        2. Entry: RSI < 35, Stoch < 30, Stoch crossover
        3. Volume: volume > 0.8 * volume_mean
        4. Volatility: 0.02 < bb_width < 0.20
        """
        conditions = (
            # Trend filter
            (dataframe["close"] > dataframe["ema_200"])
            & (dataframe["ema_50"] > dataframe["ema_100"])
            & (dataframe["adx"] > 20)
            &
            # Entry oscillator
            (dataframe["rsi"] < 35)
            & (dataframe["slowk"] < 30)
            & (dataframe["slowk"] > dataframe["slowd"])
            &
            # Volume
            (dataframe["volume"] > dataframe["volume_mean"] * 0.8)
            &
            # Volatility
            (dataframe["bb_width"] > 0.02)
            & (dataframe["bb_width"] < 0.20)
        )

        return conditions.astype(int)

    def generate_exit_signal(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Generate exit signals (1 = exit, 0 = hold).

        Logic:
        1. Overbought: RSI > 75, Stoch > 80
        2. Trend reversal: price < EMA50, MACD bearish
        """
        conditions = (
            # Overbought
            ((dataframe["rsi"] > 75) & (dataframe["slowk"] > 80))
            |
            # Trend reversal
            (
                (dataframe["close"] < dataframe["ema_50"])
                & (dataframe["macd"] < dataframe["macdsignal"])
            )
        )

        return conditions.astype(int)
