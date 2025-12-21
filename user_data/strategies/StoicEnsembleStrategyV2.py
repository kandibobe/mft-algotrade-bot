"""
Stoic Citadel - Ensemble Strategy V2 (Refactored)
==================================================

Improved version with:
1. Regime-aware behavior (adapts to market conditions)
2. Multi-strategy ensemble (momentum + mean-reversion + breakout)
3. Volume-confirmed signals
4. Time-of-day filters
5. Configurable parameters via YAML
6. Full type hints and documentation

Philosophy: "The wise man adapts to circumstances like water."

Author: Stoic Citadel Team
Version: 2.0.0
"""

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.persistence import Trade
from pandas import DataFrame
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from functools import reduce
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.utils.indicators import (
        calculate_ema, calculate_rsi, calculate_macd,
        calculate_atr, calculate_bollinger_bands,
        calculate_stochastic, calculate_adx, calculate_obv
    )
    from src.utils.regime_detection import calculate_regime_score, get_regime_parameters
    from src.utils.risk import calculate_position_size_fixed_risk, calculate_risk_metrics
    USE_CUSTOM_MODULES = True
except ImportError:
    USE_CUSTOM_MODULES = False
    import talib.abstract as ta

logger = logging.getLogger(__name__)


class StoicEnsembleStrategyV2(IStrategy):
    """
    Advanced ensemble strategy with regime detection.
    
    Combines three sub-strategies:
    1. MOMENTUM: Follow strong trends (EMA crossover + ADX)
    2. MEAN_REVERSION: Buy oversold in uptrends (RSI + BB)
    3. BREAKOUT: Enter on volatility expansion
    
    Each sub-strategy votes, and we enter when 2/3 agree.
    Risk is adjusted based on detected market regime.
    """

    # ==========================================================================
    # STRATEGY METADATA
    # ==========================================================================
    
    INTERFACE_VERSION = 3
    
    # Hyperopt spaces
    buy_rsi = IntParameter(20, 40, default=30, space="buy")
    buy_adx = IntParameter(15, 30, default=20, space="buy")
    buy_bb_width_min = DecimalParameter(0.01, 0.05, default=0.02, space="buy")
    
    sell_rsi = IntParameter(65, 85, default=75, space="sell")
    sell_hold_time = IntParameter(12, 48, default=24, space="sell")  # candles
    
    # ROI
    minimal_roi = {
        "0": 0.12,
        "30": 0.06,
        "60": 0.04,
        "120": 0.02,
        "240": 0.01
    }
    
    # Stoploss
    stoploss = -0.05
    
    # Trailing
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True
    
    # Timeframe
    timeframe = '5m'
    
    # Processing
    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count = 200
    
    # Order types
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True,
        'stoploss_on_exchange_interval': 60
    }
    
    # ==========================================================================
    # PROTECTIONS
    # ==========================================================================
    
    @property
    def protections(self):
        return [
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 60,
                "trade_limit": 3,
                "stop_duration_candles": 30,
                "required_profit": 0.0
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 288,
                "trade_limit": 5,
                "stop_duration_candles": 60,
                "max_allowed_drawdown": 0.15
            },
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 5
            }
        ]
    
    # ==========================================================================
    # INTERNAL STATE
    # ==========================================================================
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._regime_mode = 'normal'
        self._regime_params = {}
        self._last_regime_update = None
        
    # ==========================================================================
    # INDICATORS
    # ==========================================================================
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate all technical indicators.
        Uses custom vectorized implementations when available.
        """
        
        if USE_CUSTOM_MODULES:
            dataframe = self._populate_indicators_custom(dataframe)
        else:
            dataframe = self._populate_indicators_talib(dataframe)
        
        # Sub-strategy signals
        dataframe = self._calculate_momentum_signals(dataframe)
        dataframe = self._calculate_mean_reversion_signals(dataframe)
        dataframe = self._calculate_breakout_signals(dataframe)
        
        # Ensemble voting
        dataframe['ensemble_score'] = (
            dataframe['momentum_signal'].fillna(0) +
            dataframe['mean_reversion_signal'].fillna(0) +
            dataframe['breakout_signal'].fillna(0)
        )
        
        # Regime detection
        if USE_CUSTOM_MODULES and len(dataframe) >= 200:
            self._update_regime(dataframe)
        
        return dataframe
    
    def _populate_indicators_custom(self, dataframe: DataFrame) -> DataFrame:
        """Use custom vectorized indicators."""
        
        # EMAs
        dataframe['ema_9'] = calculate_ema(dataframe['close'], 9)
        dataframe['ema_21'] = calculate_ema(dataframe['close'], 21)
        dataframe['ema_50'] = calculate_ema(dataframe['close'], 50)
        dataframe['ema_100'] = calculate_ema(dataframe['close'], 100)
        dataframe['ema_200'] = calculate_ema(dataframe['close'], 200)
        
        # RSI
        dataframe['rsi'] = calculate_rsi(dataframe['close'], 14)
        dataframe['rsi_fast'] = calculate_rsi(dataframe['close'], 7)
        
        # MACD
        macd = calculate_macd(dataframe['close'])
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['signal']
        dataframe['macd_hist'] = macd['histogram']
        
        # ATR
        dataframe['atr'] = calculate_atr(
            dataframe['high'], dataframe['low'], dataframe['close'], 14
        )
        dataframe['atr_pct'] = dataframe['atr'] / dataframe['close'] * 100
        
        # Bollinger Bands
        bb = calculate_bollinger_bands(dataframe['close'], 20, 2.0)
        dataframe['bb_upper'] = bb['upper']
        dataframe['bb_middle'] = bb['middle']
        dataframe['bb_lower'] = bb['lower']
        dataframe['bb_width'] = bb['width']
        dataframe['bb_percent'] = bb['percent_b']
        
        # Stochastic
        stoch = calculate_stochastic(
            dataframe['high'], dataframe['low'], dataframe['close']
        )
        dataframe['stoch_k'] = stoch['k']
        dataframe['stoch_d'] = stoch['d']
        
        # ADX
        adx = calculate_adx(
            dataframe['high'], dataframe['low'], dataframe['close']
        )
        dataframe['adx'] = adx['adx']
        dataframe['plus_di'] = adx['plus_di']
        dataframe['minus_di'] = adx['minus_di']
        
        # OBV
        dataframe['obv'] = calculate_obv(dataframe['close'], dataframe['volume'])
        dataframe['obv_ema'] = calculate_ema(dataframe['obv'], 20)
        
        # Volume
        dataframe['volume_sma'] = dataframe['volume'].rolling(20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        
        return dataframe
    
    def _populate_indicators_talib(self, dataframe: DataFrame) -> DataFrame:
        """Fallback to TA-Lib indicators."""
        
        # EMAs
        dataframe['ema_9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=7)
        
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']
        dataframe['macd_hist'] = macd['macdhist']
        
        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_pct'] = dataframe['atr'] / dataframe['close'] * 100
        
        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['bb_width'] = (
            (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        )
        dataframe['bb_percent'] = (
            (dataframe['close'] - dataframe['bb_lower']) /
            (dataframe['bb_upper'] - dataframe['bb_lower'])
        )
        
        # Stochastic
        stoch = ta.STOCH(dataframe)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']
        
        # ADX
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=14)
        
        # Volume
        dataframe['volume_sma'] = dataframe['volume'].rolling(20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        
        return dataframe
    
    # ==========================================================================
    # SUB-STRATEGIES
    # ==========================================================================
    
    def _calculate_momentum_signals(self, dataframe: DataFrame) -> DataFrame:
        """
        Momentum sub-strategy: Follow strong trends.
        
        Entry when:
        - Price > EMA200 (uptrend)
        - EMA9 > EMA21 (short-term momentum)
        - ADX > 25 (strong trend)
        - MACD histogram positive and increasing
        """
        dataframe['momentum_signal'] = (
            (dataframe['close'] > dataframe['ema_200']) &
            (dataframe['ema_9'] > dataframe['ema_21']) &
            (dataframe['adx'] > self.buy_adx.value) &
            (dataframe['macd_hist'] > 0) &
            (dataframe['macd_hist'] > dataframe['macd_hist'].shift(1)) &
            (dataframe['plus_di'] > dataframe['minus_di'])
        ).astype(int)
        
        return dataframe
    
    def _calculate_mean_reversion_signals(self, dataframe: DataFrame) -> DataFrame:
        """
        Mean reversion sub-strategy: Buy oversold in uptrends.
        
        Entry when:
        - Price > EMA200 (still in uptrend)
        - RSI < 30 (oversold)
        - Price near lower BB
        - Stochastic crossover from oversold
        """
        dataframe['mean_reversion_signal'] = (
            (dataframe['close'] > dataframe['ema_200']) &
            (dataframe['rsi'] < self.buy_rsi.value) &
            (dataframe['bb_percent'] < 0.2) &
            (dataframe['stoch_k'] < 30) &
            (dataframe['stoch_k'] > dataframe['stoch_d']) &
            (dataframe['stoch_k'].shift(1) <= dataframe['stoch_d'].shift(1))
        ).astype(int)
        
        return dataframe
    
    def _calculate_breakout_signals(self, dataframe: DataFrame) -> DataFrame:
        """
        Breakout sub-strategy: Enter on volatility expansion.
        
        Entry when:
        - BB width expanding (volatility breakout)
        - Price breaks above recent resistance
        - Volume confirmation
        """
        # Recent high (resistance)
        dataframe['resistance'] = dataframe['high'].rolling(20).max()
        
        # BB width expanding
        dataframe['bb_width_expanding'] = (
            dataframe['bb_width'] > dataframe['bb_width'].rolling(10).mean()
        )
        
        dataframe['breakout_signal'] = (
            (dataframe['close'] > dataframe['resistance'].shift(1)) &
            (dataframe['bb_width'] > self.buy_bb_width_min.value) &
            dataframe['bb_width_expanding'] &
            (dataframe['volume_ratio'] > 1.5) &
            (dataframe['adx'] > 20)
        ).astype(int)
        
        return dataframe
    
    # ==========================================================================
    # REGIME DETECTION
    # ==========================================================================
    
    def _update_regime(self, dataframe: DataFrame) -> None:
        """
        Update market regime and adjust parameters.
        """
        try:
            regime_data = calculate_regime_score(
                dataframe['high'],
                dataframe['low'],
                dataframe['close'],
                dataframe['volume']
            )
            
            current_score = regime_data['regime_score'].iloc[-1]
            
            self._regime_params = get_regime_parameters(
                current_score,
                base_risk=0.02
            )
            
            self._regime_mode = self._regime_params.get('mode', 'normal')
            self._last_regime_update = datetime.now()
            
            logger.info(
                f"Regime updated: {self._regime_mode} "
                f"(score: {current_score:.1f})"
            )
            
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            self._regime_mode = 'normal'
    
    # ==========================================================================
    # ENTRY LOGIC
    # ==========================================================================
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define entry conditions using ensemble voting.
        
        Entry when:
        - At least 2 of 3 sub-strategies agree
        - Volume confirms
        - Not in extreme volatility
        - Time filter passes (optional)
        """
        
        # Base conditions
        base_conditions = [
            # Trend filter
            (dataframe['close'] > dataframe['ema_200']),
            (dataframe['ema_50'] > dataframe['ema_100']),
            
            # Ensemble: at least 2/3 strategies agree
            (dataframe['ensemble_score'] >= 2),
            
            # Volume confirmation
            (dataframe['volume_ratio'] > 0.8),
            
            # Volatility filter
            (dataframe['bb_width'] > self.buy_bb_width_min.value),
            (dataframe['bb_width'] < 0.15),
            
            # Not already overbought
            (dataframe['rsi'] < 70),
        ]
        
        # Regime-adjusted conditions
        if self._regime_mode == 'defensive':
            # More conservative in defensive mode
            base_conditions.append(dataframe['adx'] > 25)
            base_conditions.append(dataframe['rsi'] < 40)
        elif self._regime_mode == 'aggressive':
            # More permissive in aggressive mode
            pass  # Use base conditions
        
        # Combine conditions
        if base_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, base_conditions),
                'enter_long'
            ] = 1
        
        return dataframe
    
    # ==========================================================================
    # EXIT LOGIC
    # ==========================================================================
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define exit conditions.
        
        Exit when:
        - RSI overbought
        - Trend reversal signals
        - Time-based exit
        """
        
        conditions = []
        
        # Overbought exit
        overbought = (
            (dataframe['rsi'] > self.sell_rsi.value) &
            (dataframe['stoch_k'] > 80)
        )
        conditions.append(overbought)
        
        # Trend reversal
        trend_reversal = (
            (dataframe['close'] < dataframe['ema_50']) &
            (dataframe['macd_hist'] < 0) &
            (dataframe['macd_hist'] < dataframe['macd_hist'].shift(1))
        )
        conditions.append(trend_reversal)
        
        # Momentum loss
        momentum_loss = (
            (dataframe['ema_9'] < dataframe['ema_21']) &
            (dataframe['adx'] < 20)
        )
        conditions.append(momentum_loss)
        
        # Combine with OR logic
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'exit_long'
            ] = 1
        
        return dataframe
    
    # ==========================================================================
    # CUSTOM METHODS
    # ==========================================================================
    
    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> float:
        """
        Dynamic position sizing based on:
        1. Volatility (ATR)
        2. Current regime
        3. Risk per trade setting
        """
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return proposed_stake
        
        current = dataframe.iloc[-1]
        
        # Get ATR-based volatility
        atr_pct = current.get('atr_pct', 2.0)
        
        # Base adjustment
        if atr_pct > 5:  # High volatility
            vol_factor = 0.5
        elif atr_pct > 3:
            vol_factor = 0.75
        else:
            vol_factor = 1.0
        
        # Regime adjustment
        regime_factor = self._regime_params.get('risk_per_trade', 0.02) / 0.02
        
        # Calculate adjusted stake
        adjusted_stake = proposed_stake * vol_factor * regime_factor
        
        # Apply bounds
        adjusted_stake = max(min_stake or 0, min(adjusted_stake, max_stake))
        
        logger.debug(
            f"Position sizing: {pair} | Vol factor: {vol_factor:.2f} | "
            f"Regime factor: {regime_factor:.2f} | Stake: {adjusted_stake:.2f}"
        )
        
        return adjusted_stake
    
    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> bool:
        """
        Final validation before entry.
        
        Checks:
        - Time-of-day filter
        - Recent performance
        - Regime compatibility
        """
        
        # Time filter: avoid low liquidity hours
        hour = current_time.hour
        if hour in [0, 1, 2, 3, 4, 5]:  # Low liquidity
            if self._regime_mode != 'aggressive':
                logger.info(f"Skipping {pair}: Low liquidity hours")
                return False
        
        # In defensive mode, require stronger signals
        if self._regime_mode == 'defensive':
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty:
                current = dataframe.iloc[-1]
                if current.get('ensemble_score', 0) < 2:
                    logger.info(f"Skipping {pair}: Weak signal in defensive mode")
                    return False
        
        return True
    
    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> Optional[str]:
        """
        Custom exit logic.
        
        Implements:
        - Time-based exit
        - Profit protection
        - Emergency exit
        """
        
        # Trade duration in hours
        duration_hours = (current_time - trade.open_date_utc).total_seconds() / 3600
        
        # Emergency exit: 24h+ and losing
        if duration_hours > 24 and current_profit < -0.02:
            return "emergency_24h"
        
        # Time-based exit: 12h+ with small profit
        if duration_hours > 12 and 0 < current_profit < 0.01:
            return "timeout_small_profit"
        
        # Protect profits: if was up 5%+ and now dropping
        if hasattr(trade, 'max_rate') and trade.max_rate:
            max_profit = (trade.max_rate - trade.open_rate) / trade.open_rate
            if max_profit > 0.05 and current_profit < max_profit * 0.5:
                return "profit_protection"
        
        # Quick take profit on strong moves
        if current_profit > 0.08 and duration_hours < 2:
            return "quick_profit"
        
        return None
    
    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> float:
        """
        Dynamic leverage based on regime.
        """
        
        if self._regime_mode == 'defensive':
            return min(1.0, max_leverage)
        elif self._regime_mode == 'aggressive':
            return min(2.0, max_leverage)
        else:
            return min(1.5, max_leverage)
