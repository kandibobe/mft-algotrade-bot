"""
Stoic Citadel - Professional Ensemble Strategy
===============================================

Professional-grade ensemble strategy with:
1. Multi-strategy ensemble (momentum + mean-reversion + breakout)
2. Market regime detection and adaptation
3. Advanced risk management with correlation control
4. Dynamic position sizing based on volatility and regime
5. ML-ready architecture for signal validation

Philosophy: "The wise trader adapts to market conditions."

Author: Stoic Citadel Team
Version: 2.0.0
License: MIT
"""

from __future__ import annotations

import sys
from pathlib import Path
from functools import reduce
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame, Series
import pandas as pd
import numpy as np

# Add src to path for shared libraries
src_path = Path(__file__).parents[2] / 'src'
sys.path.insert(0, str(src_path))

try:
    from utils.regime_detection import calculate_regime_score, get_regime_parameters
    from risk.correlation import CorrelationManager
    USE_SHARED_LIBS = True
except ImportError:
    USE_SHARED_LIBS = False

import talib.abstract as ta

logger = logging.getLogger(__name__)


class StoicEnsembleStrategy(IStrategy):
    """
    Professional ensemble strategy with regime adaptation and multi-strategy voting.

    The Strategy:
    -------------
    1. THREE SUB-STRATEGIES:
       - Momentum: Follow strong trends (EMA crossover + ADX)
       - Mean Reversion: Buy oversold in uptrends (RSI + BB)
       - Breakout: Enter on volatility expansion
    
    2. ENSEMBLE VOTING: Enter when 2/3 strategies agree
    
    3. REGIME ADAPTATION: Adjust parameters based on market conditions
    
    4. ADVANCED RISK: Correlation control, portfolio heat monitoring

    Timeframe: 5m (execution)
    """

    # ==========================================================================
    # STRATEGY METADATA
    # ==========================================================================

    INTERFACE_VERSION = 3

    # Hyperopt parameters
    buy_rsi = IntParameter(20, 40, default=30, space="buy")
    buy_adx = IntParameter(15, 30, default=20, space="buy")
    sell_rsi = IntParameter(65, 85, default=75, space="sell")
    volatility_threshold = DecimalParameter(0.02, 0.10, default=0.05, space="buy")
    ensemble_threshold = IntParameter(2, 3, default=2, space="buy")  # 2/3 strategies must agree

    # Minimal ROI with regime adaptation
    minimal_roi = {
        "0": 0.15,   # 15% profit - close immediately
        "30": 0.08,  # 8% profit after 30 minutes
        "60": 0.05,  # 5% profit after 1 hour
        "120": 0.03, # 3% profit after 2 hours
        "240": 0.01  # 1% profit after 4 hours
    }

    # Stoploss - HARD LIMIT (The Stoic Guard)
    stoploss = -0.05  # -5% maximum loss per trade

    # Trailing stop - Lock in profits
    trailing_stop = True
    trailing_stop_positive = 0.01      # Start trailing at +1%
    trailing_stop_positive_offset = 0.015  # Trail by 1.5%
    trailing_only_offset_is_reached = True

    # Timeframes
    timeframe = '5m'

    # Run "populate_indicators()" only for new candles
    process_only_new_candles = True

    # Don't use signals from previous candles
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.0
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Optional order type mapping
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True,
        'stoploss_on_exchange_interval': 60
    }

    # Optional order time in force
    order_time_in_force = {
        'entry': 'GTC',  # Good Till Cancelled
        'exit': 'GTC'
    }

    # Maximum open trades based on regime
    max_open_trades = 5

    # ==========================================================================
    # INITIALIZATION
    # ==========================================================================

    def __init__(self, config: Dict) -> None:
        """Initialize strategy with regime detection and correlation management."""
        super().__init__(config)

        # Regime detection state
        self._regime_mode = 'normal'
        self._regime_params = {}
        self._last_regime_update = None
        self._regime_score = 50.0

        # Correlation manager if available
        if USE_SHARED_LIBS:
            self.correlation_manager = CorrelationManager(
                correlation_window=24,
                max_correlation=0.7,
                max_portfolio_heat=0.15
            )
        else:
            self.correlation_manager = None

        # Ensemble state
        self._ensemble_votes = {}  # Store votes per pair

        logger.info("✅ StoicEnsembleStrategy initialized with regime adaptation")

    # ==========================================================================
    # PROTECTIONS (The Stoic Cooldown)
    # ==========================================================================

    @property
    def protections(self):
        """
        Protection mechanisms with regime adaptation.
        """
        base_protections = [
            {
                # Stop trading after 3 consecutive losses
                "method": "StoplossGuard",
                "lookback_period_candles": 60,  # Last 5 hours (5m tf)
                "trade_limit": 3,
                "stop_duration_candles": 24,    # Cooldown: 2 hours
                "required_profit": 0.0
            },
            {
                # Prevent trading on pairs with low win rate
                "method": "LowProfitPairs",
                "lookback_period_candles": 360,  # Last 30 hours
                "trade_limit": 2,
                "stop_duration_candles": 60,     # Cooldown: 5 hours
                "required_profit": -0.05         # If losing > 5%
            },
            {
                # Max drawdown protection
                "method": "MaxDrawdown",
                "lookback_period_candles": 288,  # Last 24 hours
                "trade_limit": 5,
                "stop_duration_candles": 48,     # Cooldown: 4 hours
                "max_allowed_drawdown": 0.15     # 15% max DD
            }
        ]

        # Add regime-specific protections
        if self._regime_mode == 'defensive':
            base_protections.append({
                "method": "CooldownPeriod",
                "stop_duration_candles": 10
            })

        return base_protections

    # ==========================================================================
    # INDICATOR CALCULATION
    # ==========================================================================

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate all technical indicators needed for the strategy.

        This runs on EVERY candle, so keep it efficient!
        """
        # Calculate base indicators
        dataframe = self._calculate_base_indicators(dataframe)
        
        # Calculate sub-strategy signals
        dataframe = self._calculate_ensemble_signals(dataframe)
        
        # Update regime if we have enough data
        if len(dataframe) >= 200 and USE_SHARED_LIBS:
            self._update_regime(dataframe, metadata)
        
        return dataframe

    def _calculate_base_indicators(self, dataframe: DataFrame) -> DataFrame:
        """Calculate base technical indicators."""
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
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_pct'] = dataframe['atr'] / dataframe['close'] * 100
        
        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2)
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
        dataframe['volume_mean'] = dataframe['volume'].rolling(20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean']
        
        # Price changes
        dataframe['pct_change_1'] = dataframe['close'].pct_change(1)
        dataframe['pct_change_3'] = dataframe['close'].pct_change(3)
        dataframe['pct_change_5'] = dataframe['close'].pct_change(5)
        
        return dataframe

    def _calculate_ensemble_signals(self, dataframe: DataFrame) -> DataFrame:
        """Calculate signals for each sub-strategy."""
        # Momentum strategy
        dataframe['momentum_signal'] = (
            (dataframe['close'] > dataframe['ema_200']) &
            (dataframe['ema_9'] > dataframe['ema_21']) &
            (dataframe['adx'] > self.buy_adx.value) &
            (dataframe['macd_hist'] > 0) &
            (dataframe['macd_hist'] > dataframe['macd_hist'].shift(1)) &
            (dataframe['plus_di'] > dataframe['minus_di'])
        ).astype(int)
        
        # Mean reversion strategy
        dataframe['mean_reversion_signal'] = (
            (dataframe['close'] > dataframe['ema_200']) &
            (dataframe['rsi'] < self.buy_rsi.value) &
            (dataframe['bb_percent'] < 0.2) &
            (dataframe['stoch_k'] < 30) &
            (dataframe['stoch_k'] > dataframe['stoch_d']) &
            (dataframe['stoch_k'].shift(1) <= dataframe['stoch_d'].shift(1))
        ).astype(int)
        
        # Breakout strategy
        dataframe['resistance'] = dataframe['high'].rolling(20).max()
        dataframe['bb_width_expanding'] = (
            dataframe['bb_width'] > dataframe['bb_width'].rolling(10).mean()
        )
        
        dataframe['breakout_signal'] = (
            (dataframe['close'] > dataframe['resistance'].shift(1)) &
            (dataframe['bb_width'] > self.volatility_threshold.value) &
            dataframe['bb_width_expanding'] &
            (dataframe['volume_ratio'] > 1.5) &
            (dataframe['adx'] > 20)
        ).astype(int)
        
        # Ensemble voting
        dataframe['ensemble_score'] = (
            dataframe['momentum_signal'].fillna(0) +
            dataframe['mean_reversion_signal'].fillna(0) +
            dataframe['breakout_signal'].fillna(0)
        )
        
        return dataframe

    # ==========================================================================
    # ENTRY LOGIC (Ensemble Voting)
    # ==========================================================================

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define entry conditions using ensemble voting.

        Philosophy: "Enter when multiple strategies agree."
        """
        # Base conditions
        base_conditions = [
            # Trend filter
            (dataframe['close'] > dataframe['ema_200']),
            (dataframe['ema_50'] > dataframe['ema_100']),
            
            # Ensemble: required number of strategies agree
            (dataframe['ensemble_score'] >= self.ensemble_threshold.value),
            
            # Volume confirmation
            (dataframe['volume_ratio'] > 0.8),
            
            # Volatility filter
            (dataframe['bb_width'] > self.volatility_threshold.value),
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
        
        # Store votes for this pair
        self._ensemble_votes[metadata['pair']] = {
            'momentum': dataframe['momentum_signal'].sum(),
            'mean_reversion': dataframe['mean_reversion_signal'].sum(),
            'breakout': dataframe['breakout_signal'].sum(),
            'ensemble': dataframe['ensemble_score'].sum(),
            'entries': dataframe['enter_long'].sum()
        }
        
        if dataframe['enter_long'].sum() > 0:
            logger.info(
                f"📊 {metadata['pair']}: {dataframe['enter_long'].sum()} entry signals "
                f"(regime: {self._regime_mode}, votes: {dataframe['ensemble_score'].iloc[-1]}/3)"
            )
        
        return dataframe

    # ==========================================================================
    # EXIT LOGIC (Regime-Adaptive)
    # ==========================================================================

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define exit conditions with regime adaptation.

        Philosophy: "Exit when the market tells you to."
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
        
        # Regime-specific exits
        if self._regime_mode == 'defensive':
            # Exit earlier in defensive mode
            defensive_exit = (
                (dataframe['rsi'] > self.sell_rsi.value - 5) |
                (dataframe['close'] < dataframe['ema_50'])
            )
            conditions.append(defensive_exit)
        elif self._regime_mode == 'aggressive':
            # Hold longer in aggressive mode
            aggressive_exit = (
                (dataframe['rsi'] > self.sell_rsi.value + 5) |
                ((dataframe['close'] < dataframe['ema_50']) & (dataframe['macd'] < dataframe['macdsignal']))
            )
            conditions.append(aggressive_exit)
        
        # Combine with OR logic
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'exit_long'
            ] = 1
        
        exit_count = dataframe['exit_long'].sum()
        if exit_count > 0:
            logger.info(
                f"📉 {metadata['pair']}: {exit_count} exit signals "
                f"(regime: {self._regime_mode})"
            )
        
        return dataframe

    # ==========================================================================
    # HELPER METHODS
    # ==========================================================================

    def _update_regime(self, dataframe: DataFrame, metadata: dict) -> None:
        """Update market regime based on current data."""
        try:
            regime_data = calculate_regime_score(
                dataframe['high'],
                dataframe['low'],
                dataframe['close'],
                dataframe['volume']
            )
            
            current_score = regime_data['regime_score'].iloc[-1]
            self._regime_score = float(current_score)
            
            self._regime_params = get_regime_parameters(
                current_score,
                base_risk=0.02
            )
            
            self._regime_mode = self._regime_params.get('mode', 'normal')
            self._last_regime_update = datetime.now()
            
            logger.info(
                f"📈 Regime updated: {self._regime_mode} "
                f"(score: {current_score:.1f}) for {metadata['pair']}"
            )
            
        except Exception as e:
            logger.warning(f"⚠️ Regime detection failed: {e}")
            self._regime_mode = 'normal'

    # ==========================================================================
    # CUSTOM METHODS (Enhanced with Regime Adaptation)
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
        Dynamic position sizing based on volatility and regime.

        Higher volatility = smaller position (risk parity).
        Regime adaptation adjusts risk per trade.
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty:
                return proposed_stake
                
            current_candle = dataframe.iloc[-1].squeeze()

            # Get ATR (volatility measure)
            atr = current_candle.get('atr', 0)
            close = current_candle.get('close', current_rate)

            # Calculate volatility percentage
            volatility_pct = atr / close if close > 0 else 0.02

            # Base volatility adjustment
            if volatility_pct > self.volatility_threshold.value:  # High volatility
                vol_factor = 0.5
                vol_msg = "High volatility, reducing stake by 50%"
            elif volatility_pct > self.volatility_threshold.value * 0.6:  # Medium volatility
                vol_factor = 0.75
                vol_msg = "Medium volatility, reducing stake by 25%"
            else:  # Low volatility
                vol_factor = 1.0
                vol_msg = "Low volatility, full stake"

            # Regime adjustment
            regime_factor = self._regime_params.get('risk_per_trade', 0.02) / 0.02

            # Calculate adjusted stake
            adjusted_stake = proposed_stake * vol_factor * regime_factor

            # Apply bounds
            adjusted_stake = max(min_stake or 0, min(adjusted_stake, max_stake))

            logger.info(
                f"💰 {pair}: {vol_msg} | "
                f"Regime factor: {regime_factor:.2f} | "
                f"Final stake: {adjusted_stake:.2f}"
            )

            return adjusted_stake

        except Exception as e:
            logger.error(f"❌ Error calculating stake: {e}")
            return proposed_stake

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
        Final gate before entering a trade with correlation check.
        """
        # 1. Check low liquidity hours
        hour = current_time.hour
        if hour in [0, 1, 2, 3, 4, 5]:
            logger.info(f"⏰ {pair}: Rejecting entry - low liquidity hours")
            return False

        # 2. Correlation check if manager available
        if self.correlation_manager:
            try:
                open_trades = self.dp.get_open_trades()
                if open_trades:
                    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                    
                    all_pairs_data = {}
                    for trade in open_trades:
                        open_pair = trade['pair']
                        open_df, _ = self.dp.get_analyzed_dataframe(
                            open_pair,
                            self.timeframe
                        )
                        all_pairs_data[open_pair] = open_df

                    correlation_ok = self.correlation_manager.check_entry_correlation(
                        new_pair=pair,
                        new_pair_data=dataframe,
                        open_positions=[{'pair': t['pair']} for t in open_trades],
                        all_pairs_data=all_pairs_data
                    )

                    if not correlation_ok:
                        logger.warning(f"❌ {pair}: Entry blocked by correlation check")
                        return False
            except Exception as e:
                logger.warning(f"⚠️ Correlation check failed: {e}, allowing entry")

        # 3. Regime-specific checks
        if self._regime_mode == 'defensive':
            # In defensive mode, require stronger ensemble agreement
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty:
                current_score = dataframe['ensemble_score'].iloc[-1]
                if current_score < 3:  # Require all 3 strategies in defensive mode
                    logger.info(f"⚠️ {pair}: Weak ensemble signal in defensive mode")
                    return False

        logger.info(f"✅ {pair}: Entry confirmed (passed all checks)")
        return True

    def custom_exit(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> Optional[str]:
        """
        Custom exit logic with regime adaptation and emergency exits.
        """
        # Trade duration in hours
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 3600

        # Emergency exit: trade open > 24h and losing
        if trade_duration > 24 and current_profit < -0.02:
            logger.warning(f"⚠️ {pair}: Emergency exit - open 24h+ with -2% loss")
            return "emergency_exit_24h"

        # Regime-specific exits
        if self._regime_mode == 'defensive':
            # Exit earlier in defensive mode
            if trade_duration > 12 and current_profit > 0.03:
                logger.info(f"💰 {pair}: Early take profit at +3% in defensive mode")
                return "early_take_profit_defensive"
                
            if trade_duration > 6 and current_profit < -0.01:
                logger.info(f"📉 {pair}: Quick stop loss at -1% in defensive mode")
                return "quick_stop_defensive"

        elif self._regime_mode == 'aggressive':
            # Hold longer in aggressive mode
            if current_profit > 0.15:
                logger.info(f"💰 {pair}: Take profit at +15% in aggressive mode")
                return "take_profit_aggressive"
                
            if trade_duration > 36 and current_profit < -0.03:
                logger.warning(f"⚠️ {pair}: Extended emergency exit in aggressive mode")
                return "extended_emergency_aggressive"

        else:  # normal/cautious
            # Standard exits
            if current_profit > 0.10:
                logger.info(f"💰 {pair}: Take profit at +10%")
                return "take_profit_10pct"
                
            if trade_duration > 18 and 0 < current_profit < 0.02:
                logger.info(f"⏰ {pair}: Time-based exit with small profit")
                return "time_exit_small_profit"

        # Protect profits: if was up 5%+ and now dropping
        if hasattr(trade, 'max_rate') and trade.max_rate:
            max_profit = (trade.max_rate - trade.open_rate) / trade.open_rate
            if max_profit > 0.05 and current_profit < max_profit * 0.5:
                logger.info(f"🛡️ {pair}: Profit protection exit")
                return "profit_protection"

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
        Dynamic leverage based on regime and volatility.
        """
        # Get current volatility
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty:
                current = dataframe.iloc[-1]
                atr_pct = current.get('atr_pct', 2.0)
                
                # Reduce leverage in high volatility
                if atr_pct > 5:
                    vol_factor = 0.5
                elif atr_pct > 3:
                    vol_factor = 0.75
                else:
                    vol_factor = 1.0
            else:
                vol_factor = 1.0
        except:
            vol_factor = 1.0

        # Regime-based leverage
        if self._regime_mode == 'defensive':
            regime_leverage = min(1.0, max_leverage)
        elif self._regime_mode == 'aggressive':
            regime_leverage = min(2.0, max_leverage)
        else:
            regime_leverage = min(1.5, max_leverage)

        # Apply both adjustments
        final_leverage = min(regime_leverage * vol_factor, max_leverage)
        
        logger.info(
            f"⚖️ {pair}: Leverage - Regime: {regime_leverage:.1f}x, "
            f"Vol factor: {vol_factor:.2f}, Final: {final_leverage:.1f}x"
        )
        
        return final_leverage
