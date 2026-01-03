"""
Stoic Citadel - Core Trading Logic
==================================

Pure logic layer for trading decisions.
Decoupled from Freqtrade to allow independent testing and simulation.
Supports both scalar (unit test) and vectorized (backtest) operations.

Refactored V6: Regime-Permission Matrix + Signal Persistence
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import logging

# Import indicators and regime
from src.utils.indicators import (
    calculate_ema, calculate_rsi, calculate_macd,
    calculate_atr, calculate_bollinger_bands,
    calculate_stochastic, calculate_adx, calculate_obv
)
from src.utils.regime_detection import calculate_regime, MarketRegime
from src.utils.logger import log, log_strategy_signal

logger = logging.getLogger(__name__)

@dataclass
class StructuredTradeDecision:
    signal: str  # 'buy', 'sell', 'hold'
    confidence: float
    regime: str
    reason: str
    metadata: Dict[str, Any]

    @property
    def should_enter_long(self) -> bool:
        return self.signal == 'buy'

    @property
    def should_exit_long(self) -> bool:
        return self.signal == 'sell'

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
        Calculate technical indicators required for the strategy.
        """
        # Basic needed for logic
        dataframe['ema_50'] = calculate_ema(dataframe['close'], 50)
        dataframe['ema_200'] = calculate_ema(dataframe['close'], 200)
        dataframe['rsi'] = calculate_rsi(dataframe['close'], 14)
        
        macd = calculate_macd(dataframe['close'])
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['signal']
        dataframe['macd_hist'] = macd['histogram']

        bb = calculate_bollinger_bands(dataframe['close'], 20, 2.0)
        dataframe['bb_lower'] = bb['lower']
        dataframe['bb_middle'] = bb['middle']
        dataframe['bb_upper'] = bb['upper']
        dataframe['bb_width'] = bb['width']
        
        # Calculate Regime Indicators (if not already done by strategy)
        # We need ATR for Regime Detection
        # regime_detection.calculate_regime handles indicators internally
        # but we might need to merge results back if strategy doesn't do it.
        # For simplicity, we assume strategy calls calculate_regime and merges.
        
        return dataframe
    
    @staticmethod
    def populate_entry_exit_signals(dataframe: pd.DataFrame, 
                                  buy_threshold: float = 0.6,
                                  sell_rsi: int = 75,
                                  mean_rev_rsi: int = 30,
                                  persistence_window: int = 3) -> pd.DataFrame:
        """
        Vectorized signal generation using Regime Permission Matrix.
        
        Returns dataframe with 'enter_long' and 'exit_long' columns.
        """
        df = dataframe.copy()
        df['enter_long'] = 0
        df['exit_long'] = 0
        
        # Ensure Regime is calculated
        if 'regime' not in df.columns:
            # Fallback calculation (computationally expensive)
            logger.warning("Regime column missing, calculating on the fly.")
            regime_df = calculate_regime(df['high'], df['low'], df['close'], df['volume'])
            df['regime'] = regime_df['regime']
            df['vol_zscore'] = regime_df['vol_zscore']

        # --- Signal Generators (Raw) ---
        
        # 1. Trend Signal (Breakout/Momentum)
        # Condition: Price > EMA200 AND ML > Threshold
        raw_trend_signal = (
            (df['close'] > df['ema_200']) & 
            (df['ml_prediction'] > buy_threshold)
        )
        
        # 2. Mean Reversion Signal (Dip Buy)
        # Condition: RSI < mean_rev_rsi AND Price < BB Lower
        raw_mean_rev_signal = (
            (df['rsi'] < mean_rev_rsi) &
            (df['close'] <= df['bb_lower'])
        )
        
        # --- Signal Persistence (Glitch Filter) ---
        # Signal must be true for N periods to be valid
        # We use rolling min() on boolean (converted to int)
        # If min is 1, then it was true for all N periods.
        # FIX: Use threshold > 0.99 instead of == 1 to avoid float comparison issues
        trend_persistent = raw_trend_signal.astype(int).rolling(persistence_window).min() > 0.99
        mean_rev_persistent = raw_mean_rev_signal.astype(int).rolling(persistence_window).min() > 0.99
        
        # --- Regime Permission Matrix ---
        
        # 1. PUMP_DUMP (High Vol + Trend) -> ALLOW TREND, BAN MEAN REV
        mask_pump = df['regime'] == MarketRegime.PUMP_DUMP.value
        df.loc[mask_pump & trend_persistent, 'enter_long'] = 1
        
        # 2. GRIND (Low Vol + Trend) -> ALLOW TREND (Accumulate), BAN MEAN REV
        mask_grind = df['regime'] == MarketRegime.GRIND.value
        # In grind, we might be less strict on persistence or require smaller size
        df.loc[mask_grind & trend_persistent, 'enter_long'] = 1
        
        # 3. VIOLENT_CHOP (High Vol + Range) -> ALLOW MEAN REV, BAN TREND
        mask_violent = df['regime'] == MarketRegime.VIOLENT_CHOP.value
        df.loc[mask_violent & mean_rev_persistent, 'enter_long'] = 1
        
        # 4. QUIET_CHOP (Low Vol + Range) -> STAY FLAT
        # No assignments. Default is 0.
        
        # --- Exit Logic ---
        # Standard RSI Overbought Exit
        # In High Vol, we might want to exit earlier or trail tighter
        exit_cond = (df['rsi'] > sell_rsi)
        df.loc[exit_cond, 'exit_long'] = 1
        
        return df

    @staticmethod
    def get_entry_decision(candle: Dict[str, Any], regime: MarketRegime, threshold: float = 0.6) -> StructuredTradeDecision:
        """
        Scalar version for unit testing or event-driven execution.
        """
        # Extract features
        rsi = candle.get('rsi', 50)
        close = candle.get('close', 0)
        bb_lower = candle.get('bb_lower', 0)
        ema_200 = candle.get('ema_200', 0)
        ml_pred = candle.get('ml_prediction', 0.5)
        symbol = candle.get('symbol', 'unknown')

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
                "threshold": threshold
            }
        )

        # Log significant decisions (Entry or Exit)
        if decision.should_enter_long or decision.should_exit_long:
            log_strategy_signal(
                strategy="StoicLogic",
                symbol=symbol,
                signal=decision.signal,
                confidence=decision.confidence,
                indicators=decision.metadata,
                reason=decision.reason
            )

        return decision
