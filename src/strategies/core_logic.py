"""
Stoic Citadel - Core Trading Logic
==================================

Pure logic layer for trading decisions.
Decoupled from Freqtrade to allow independent testing and simulation.
Supports both scalar (unit test) and vectorized (backtest) operations.

Refactored V6: Regime-Permission Matrix + Signal Persistence
"""

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

# Import indicators and regime
from src.utils.indicators import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
)
from src.utils.logger import log_strategy_signal
from src.utils.regime_detection import MarketRegime, calculate_regime

logger = logging.getLogger(__name__)


@dataclass
class StructuredTradeDecision:
    signal: str  # 'buy', 'sell', 'hold'
    confidence: float
    regime: str
    reason: str
    metadata: dict[str, Any]

    @property
    def should_enter_long(self) -> bool:
        return self.signal == "buy"

    @property
    def should_exit_long(self) -> bool:
        return self.signal == "sell"


# Legacy Alias for backward compatibility
TradeDecision = StructuredTradeDecision


class StoicLogic:
    """
    Encapsulates the core decision making logic.
    """

    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        """Wrapper for EMA calculation."""
        return calculate_ema(series, period)

    @staticmethod
    def populate_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive technical indicator calculation.
        Includes technical indicators, column aliasing, and safety fallbacks.
        """
        df = dataframe.copy()

        # 1. Core Technicals
        df["ema_50"] = calculate_ema(df["close"], 50)
        df["ema_100"] = calculate_ema(df["close"], 100)
        df["ema_200"] = calculate_ema(df["close"], 200)
        df["rsi"] = calculate_rsi(df["close"], 14)
        df["atr"] = calculate_atr(df["high"], df["low"], df["close"], 14)

        macd = calculate_macd(df["close"])
        df["macd"] = macd["macd"]
        df["macd_signal"] = macd["signal"]
        df["macd_hist"] = macd["histogram"]

        bb = calculate_bollinger_bands(df["close"], 20, 2.0)
        df["bb_lower"] = bb["lower"]
        df["bb_middle"] = bb["middle"]
        df["bb_upper"] = bb["upper"]
        df["bb_width"] = bb["width"]

        # Stochastic (Legacy/Robustness)
        low_min = df["low"].rolling(window=14).min()
        high_max = df["high"].rolling(window=14).max()
        df["slowk"] = 100 * (df["close"] - low_min) / (high_max - low_min)
        df["slowd"] = df["slowk"].rolling(window=3).mean()

        # Volume
        df["volume_mean"] = df["volume"].rolling(window=20).mean()

        # 2. Aliases for backward compatibility and tests
        df["bb_lowerband"] = df["bb_lower"]
        df["bb_upperband"] = df["bb_upper"]
        df["bb_middleband"] = df["bb_middle"]
        df["macdsignal"] = df["macd_signal"]
        df["macdhist"] = df["macd_hist"]

        return df

    @staticmethod
    def populate_entry_exit_signals(
        dataframe: pd.DataFrame,
        buy_threshold: float = 0.6,
        sell_rsi: int = 75,
        mean_rev_rsi: int = 30,
        persistence_window: int = 3,
    ) -> pd.DataFrame:
        """
        Vectorized signal generation using Regime Permission Matrix.

        Returns dataframe with 'enter_long' and 'exit_long' columns.
        """
        df = dataframe.copy()
        df["enter_long"] = 0
        df["exit_long"] = 0

        # Ensure Regime is calculated
        if "regime" not in df.columns:
            # Fallback calculation (computationally expensive)
            logger.warning("Regime column missing, calculating on the fly.")
            regime_df = calculate_regime(df["high"], df["low"], df["close"], df["volume"])
            df["regime"] = regime_df["regime"]
            df["vol_zscore"] = regime_df["vol_zscore"]

        # --- Signal Generators (Raw) ---

        # 1. Trend Signal (Breakout/Momentum)
        # Condition: Price > EMA200 AND ML > Threshold
        raw_trend_signal = (df["close"] > df["ema_200"]) & (
            df.get("ml_prediction", 0.5) > buy_threshold
        )

        # 2. Mean Reversion Signal (Dip Buy)
        # Condition: RSI < mean_rev_rsi AND Price < BB Lower
        raw_mean_rev_signal = (df["rsi"] < mean_rev_rsi) & (df["close"] <= df["bb_lower"])

        # --- Signal Persistence (Glitch Filter) ---
        # Signal must be true for N periods to be valid
        trend_persistent = raw_trend_signal.astype(int).rolling(persistence_window).min() > 0.99
        mean_rev_persistent = (
            raw_mean_rev_signal.astype(int).rolling(persistence_window).min() > 0.99
        )

        # --- Regime Permission Matrix ---

        # 1. PUMP_DUMP (High Vol + Trend) -> ALLOW TREND, BAN MEAN REV
        mask_pump = df["regime"] == MarketRegime.PUMP_DUMP.value
        df.loc[mask_pump & trend_persistent, "enter_long"] = 1

        # 2. GRIND (Low Vol + Trend) -> ALLOW TREND (Accumulate), BAN MEAN REV
        mask_grind = df["regime"] == MarketRegime.GRIND.value
        df.loc[mask_grind & trend_persistent, "enter_long"] = 1

        # 3. VIOLENT_CHOP (High Vol + Range) -> ALLOW MEAN REV, BAN TREND
        mask_violent = df["regime"] == MarketRegime.VIOLENT_CHOP.value
        df.loc[mask_violent & mean_rev_persistent, "enter_long"] = 1

        # 4. QUIET_CHOP (Low Vol + Range) -> STAY FLAT
        # No assignments. Default is 0.

        # --- Exit Logic ---
        # Standard RSI Overbought Exit
        exit_cond = df["rsi"] > sell_rsi
        df.loc[exit_cond, "exit_long"] = 1

        return df

    @staticmethod
    def get_entry_decision(
        candle: dict[str, Any], regime: MarketRegime, threshold: float = 0.6
    ) -> StructuredTradeDecision:
        """
        Scalar version for unit testing or event-driven execution.
        """
        # Extract features
        rsi = candle.get("rsi", 50)
        close = candle.get("close", 0)
        bb_lower = candle.get("bb_lower", 0)
        ema_200 = candle.get("ema_200", 0)
        ml_pred = candle.get("ml_prediction", 0.5)
        symbol = candle.get("symbol", "unknown")

        # Default: Hold
        signal = "hold"
        reason = "No Signal"

        # Logic Matrix
        if regime == MarketRegime.QUIET_CHOP:
            reason = "Quiet Chop - Stay Flat"

        elif regime == MarketRegime.PUMP_DUMP:
            # Trend Following Allowed
            if (close > ema_200) and (ml_pred > threshold):
                signal = "buy"
                reason = "Trend Follow (Pump)"

        elif regime == MarketRegime.GRIND:
            # Trend Following Allowed
            if (close > ema_200) and (ml_pred > threshold):
                signal = "buy"
                reason = "Trend Follow (Grind)"

        elif regime == MarketRegime.VIOLENT_CHOP:
            # Mean Reversion Allowed
            if (rsi < 30) and (close <= bb_lower):
                signal = "buy"
                reason = "Mean Reversion (Violent)"

        # Construct Structured Decision
        decision = StructuredTradeDecision(
            signal=signal,
            confidence=float(ml_pred),
            regime=str(regime),
            reason=reason,
            metadata={
                "rsi": float(rsi),
                "ema_200": float(ema_200),
                "close": float(close),
                "bb_lower": float(bb_lower),
                "threshold": threshold,
            },
        )

        # Log significant decisions (Entry or Exit)
        if decision.should_enter_long or decision.should_exit_long:
            log_strategy_signal(
                strategy="StoicLogic",
                symbol=symbol,
                signal=decision.signal,
                confidence=decision.confidence,
                indicators=decision.metadata,
                reason=decision.reason,
            )

        return decision
