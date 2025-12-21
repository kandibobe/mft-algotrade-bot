"""
Stoic Citadel - Ensemble Strategy V3 (Meta-Learning Enhanced)
==============================================================

Enhanced version with meta-learning ensemble that learns optimal weights
for base models using Logistic Regression as meta-learner.

Features:
1. Meta-learning ensemble with Logistic Regression meta-learner
2. Dynamic model weighting based on historical performance
3. Confidence scores based on model agreement
4. Online learning and weight updates
5. All features from V2 (regime detection, multi-strategy, etc.)

Philosophy: "The wise man learns from experience and adapts."

Author: Stoic Citadel Team
Version: 3.0.0
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
    from src.ml.meta_learning import MetaLearningEnsemble, MetaLearningConfig
    USE_CUSTOM_MODULES = True
    META_LEARNING_AVAILABLE = True
except ImportError as e:
    USE_CUSTOM_MODULES = False
    META_LEARNING_AVAILABLE = False
    import talib.abstract as ta
    logger = logging.getLogger(__name__)
    logger.warning(f"Meta-learning not available: {e}. Falling back to simple voting.")

logger = logging.getLogger(__name__)


class StoicEnsembleStrategyV3(IStrategy):
    """
    Meta-learning enhanced ensemble strategy with regime detection.
    
    Combines three sub-strategies with learned optimal weights:
    1. MOMENTUM: Follow strong trends (EMA crossover + ADX)
    2. MEAN_REVERSION: Buy oversold in uptrends (RSI + BB)
    3. BREAKOUT: Enter on volatility expansion
    
    Uses Logistic Regression meta-learner to weight sub-strategies optimally
    based on historical performance. Provides confidence scores for risk management.
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
    
    # Meta-learning parameters
    meta_learning_threshold = DecimalParameter(0.5, 0.8, default=0.6, space="buy")
    confidence_threshold = DecimalParameter(0.3, 0.7, default=0.5, space="buy")
    
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
        
        # Meta-learning ensemble
        self.meta_ensemble = None
        self._base_models_initialized = False
        self._training_data = []
        self._training_labels = []
        self._max_training_samples = 1000
        
        # Initialize meta-learning if available
        if META_LEARNING_AVAILABLE:
            self._initialize_meta_learning()
        else:
            logger.warning("Meta-learning not available. Using simple voting ensemble.")
    
    def _initialize_meta_learning(self):
        """Initialize meta-learning ensemble with dummy base models."""
        # Create dummy base models that will be replaced with actual signal functions
        # In practice, these would be ML models, but for our strategy they're signal functions
        class DummyBaseModel:
            def predict_proba(self, X):
                # X will be the feature matrix, but we need to extract signals
                # This will be handled in _get_base_predictions method
                return np.column_stack([1 - X[:, 0], X[:, 0]])
        
        # Create 3 dummy models (for momentum, mean reversion, breakout)
        dummy_models = [DummyBaseModel() for _ in range(3)]
        
        # Configure meta-learning
        config = MetaLearningConfig(
            meta_model_path="user_data/models/meta_ensemble_v3.pkl",
            min_samples_for_training=200,
            retrain_interval=500
        )
        
        self.meta_ensemble = MetaLearningEnsemble(dummy_models, config)
        
        # Try to load pre-trained model
        if not self.meta_ensemble.load():
            logger.info("No pre-trained meta-model found. Will train when enough data is available.")
        
        logger.info("Meta-learning ensemble initialized")
    
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
        
        # Calculate signal probabilities (0-1) from binary signals
        # These will be used as base model predictions for meta-learning
        dataframe['momentum_prob'] = dataframe['momentum_signal'].astype(float)
        dataframe['mean_reversion_prob'] = dataframe['mean_reversion_signal'].astype(float)
        dataframe['breakout_prob'] = dataframe['breakout_signal'].astype(float)
        
        # Simple ensemble voting (fallback)
        dataframe['ensemble_score'] = (
            dataframe['momentum_signal'].fillna(0) +
            dataframe['mean_reversion_signal'].fillna(0) +
            dataframe['breakout_signal'].fillna(0)
        )
        
        # Meta-learning predictions if available
        if self.meta_ensemble and self.meta_ensemble.is_trained:
            try:
                # Prepare features for meta-learning (current candle)
                current_features = self._prepare_meta_features(dataframe)
                if current_features is not None and len(current_features) > 0:
                    # Get meta-learning predictions
                    predictions, confidence = self.meta_ensemble.predict_with_confidence(
                        current_features
                    )
                    # Store predictions
                    if len(predictions) == len(dataframe):
                        dataframe['meta_prediction'] = predictions
                        dataframe['meta_confidence'] = confidence
                    else:
                        # Handle mismatch
                        dataframe['meta_prediction'] = dataframe['ensemble_score'] / 3.0
                        dataframe['meta_confidence'] = 0.5
                else:
                    dataframe['meta_prediction'] = dataframe['ensemble_score'] / 3.0
                    dataframe['meta_confidence'] = 0.5
            except Exception as e:
                logger.error(f"Meta-learning prediction failed: {e}")
                dataframe['meta_prediction'] = dataframe['ensemble_score'] / 3.0
                dataframe['meta_confidence'] = 0.5
        else:
            # Fallback to simple average
            dataframe['meta_prediction'] = dataframe['ensemble_score'] / 3.0
            dataframe['meta_confidence'] = 0.5
        
        # Regime detection
        if USE_CUSTOM_MODULES and len(dataframe) >= 200:
            self._update_regime(dataframe)
        
        return dataframe
    
    def _prepare_meta_features(self, dataframe: DataFrame) -> Optional[np.ndarray]:
        """
        Prepare features for meta-learning ensemble.
        
        Returns:
            Feature matrix for current candles
        """
        try:
            # Use the last N candles for prediction
            lookback = 10
            if len(dataframe) < lookback:
                return None
            
            # Extract relevant features for the last 'lookback' candles
            features = []
            for i in range(len(dataframe)):
                start_idx = max(0, i - lookback + 1)
                window = dataframe.iloc[start_idx:i+1]
                
                # Calculate window statistics
                if len(window) > 0:
                    # Signal probabilities in the window
                    momentum_mean = window['momentum_prob'].mean()
                    mean_reversion_mean = window['mean_reversion_prob'].mean()
                    breakout_mean = window['breakout_prob'].mean()
                    
                    # Price action features
                    price_change = (window['close'].iloc[-1] / window['close'].iloc[0] - 1) if len(window) > 1 else 0
                    volatility = window['bb_width'].std() if 'bb_width' in window else 0
                    volume_ratio = window['volume_ratio'].mean() if 'volume_ratio' in window else 1.0
                    
                    # Combine features
                    feature_vector = [
                        momentum_mean,
                        mean_reversion_mean,
                        breakout_mean,
                        price_change,
                        volatility,
                        volume_ratio,
                        window['rsi'].iloc[-1] if 'rsi' in window and len(window) > 0 else 50,
                        window['adx'].iloc[-1] if 'adx' in window and len(window) > 0 else 20
                    ]
                    features.append(feature_vector)
                else:
                    # Default feature vector
                    features.append([0.5, 0.5, 0.5, 0, 0, 1.0, 50, 20])
            
            return np.array(features)
        except Exception as e:
            logger.error(f"Failed to prepare meta-features: {e}")
            return None
    
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
        Define entry conditions using meta-learning ensemble.
        
        Entry when:
        - Meta-learning prediction > threshold
        - Confidence > confidence threshold
        - Volume confirms
        - Not in extreme volatility
        """
        
        # Base conditions
        base_conditions = [
            # Trend filter
            (dataframe['close'] > dataframe['ema_200']),
            (dataframe['ema_50'] > dataframe['ema_100']),
            
            # Meta-learning prediction
            (dataframe['meta_prediction'] > self.meta_learning_threshold.value),
            
            # Confidence threshold
            (dataframe['meta_confidence'] > self.confidence_threshold.value),
            
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
            # Higher confidence required
            base_conditions.append(dataframe['meta_confidence'] > self.confidence_threshold.value + 0.1)
        elif self._regime_mode == 'aggressive':
            # More permissive in aggressive mode
            pass  # Use base conditions
        
        # Combine conditions
        if base_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, base_conditions),
                'enter_long'
            ] = 1
        
        # Log entry signals
        entry_count = dataframe['enter_long'].sum()
        if entry_count > 0:
            logger.info(
                f"📊 {metadata['pair']}: {entry_count} entry signals "
                f"(meta-pred: {dataframe['meta_prediction'].iloc[-1]:.2f}, "
                f"conf: {dataframe['meta_confidence'].iloc[-1]:.2f}, "
                f"regime: {self._regime_mode})"
            )
        
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
        
        # Meta-learning based exit (if prediction drops below threshold)
        if 'meta_prediction' in dataframe.columns:
            weak_prediction = (
                (dataframe['meta_prediction'] < 0.4) &
                (dataframe['meta_confidence'] > 0.6)
            )
            conditions.append(weak_prediction)
        
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
        3. Meta-learning confidence
        4. Risk per trade setting
        """
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return proposed_stake
        
        current = dataframe.iloc[-1]
        
        # Get ATR-based volatility
        atr_pct = current.get('atr_pct', 2.0)
        
        # Base volatility adjustment
        if atr_pct > 5:  # High volatility
            vol_factor = 0.5
        elif atr_pct > 3:
            vol_factor = 0.75
        else:
            vol_factor = 1.0
        
        # Regime adjustment
        regime_factor = self._regime_params.get('risk_per_trade', 0.02) / 0.02
        
        # Confidence-based adjustment
        confidence = current.get('meta_confidence', 0.5)
        confidence_factor = 0.5 + confidence  # 0.5-1.5 range
        
        # Calculate adjusted stake
        adjusted_stake = proposed_stake * vol_factor * regime_factor * confidence_factor
        
        # Apply bounds
        adjusted_stake = max(min_stake or 0, min(adjusted_stake, max_stake))
        
        logger.debug(
            f"Position sizing: {pair} | Vol factor: {vol_factor:.2f} | "
            f"Regime factor: {regime_factor:.2f} | "
            f"Confidence factor: {confidence_factor:.2f} | "
            f"Stake: {adjusted_stake:.2f}"
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
        - Meta-learning confidence
        """
        
        # Time filter: avoid low liquidity hours
        hour = current_time.hour
        if hour in [0, 1, 2, 3, 4, 5]:  # Low liquidity
            if self._regime_mode != 'aggressive':
                logger.info(f"Skipping {pair}: Low liquidity hours")
                return False
        
        # Get current data
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return True
        
        current = dataframe.iloc[-1]
        
        # In defensive mode, require stronger signals
        if self._regime_mode == 'defensive':
            if current.get('meta_confidence', 0) < self.confidence_threshold.value + 0.1:
                logger.info(f"Skipping {pair}: Low confidence in defensive mode")
                return False
        
        # Check meta-learning prediction
        meta_pred = current.get('meta_prediction', 0.5)
        if meta_pred < self.meta_learning_threshold.value:
            logger.info(f"Skipping {pair}: Weak meta-learning prediction")
            return False
        
        # Check confidence
        confidence = current.get('meta_confidence', 0.5)
        if confidence < self.confidence_threshold.value:
            logger.info(f"Skipping {pair}: Low confidence")
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
        - Meta-learning based exit
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
        
        # Meta-learning based exit (check current prediction)
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty:
                current = dataframe.iloc[-1]
                meta_pred = current.get('meta_prediction', 0.5)
                confidence = current.get('meta_confidence', 0.5)
                
                # Exit if prediction turns strongly negative with high confidence
                if meta_pred < 0.3 and confidence > 0.7:
                    return "meta_learning_sell_signal"
                
                # Exit if confidence drops significantly
                entry_confidence = getattr(trade, 'entry_confidence', 0.5)
                if confidence < entry_confidence * 0.5 and current_profit > 0:
                    return "confidence_drop"
        except:
            pass
        
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
        Dynamic leverage based on regime and confidence.
        """
        
        # Get current data for confidence
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty:
                current = dataframe.iloc[-1]
                confidence = current.get('meta_confidence', 0.5)
            else:
                confidence = 0.5
        except:
            confidence = 0.5
        
        # Base leverage based on regime
        if self._regime_mode == 'defensive':
            base_leverage = min(1.0, max_leverage)
        elif self._regime_mode == 'aggressive':
            base_leverage = min(2.0, max_leverage)
        else:
            base_leverage = min(1.5, max_leverage)
        
        # Adjust based on confidence
        confidence_factor = 0.5 + confidence  # 0.5-1.5 range
        final_leverage = min(base_leverage * confidence_factor, max_leverage)
        
        logger.debug(
            f"Leverage: {pair} | Regime: {self._regime_mode} | "
            f"Confidence: {confidence:.2f} | Final: {final_leverage:.1f}x"
        )
        
        return final_leverage
    
    def collect_training_data(self, pair: str, success: bool):
        """
        Collect training data for meta-learning ensemble.
        
        This should be called after trades close to collect
        features and labels for training.
        """
        if not self.meta_ensemble:
            return
        
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty or len(dataframe) < 20:
                return
            
            # Get features from the entry point (last 20 candles before entry)
            entry_features = self._prepare_meta_features(dataframe.iloc[-20:])
            if entry_features is not None and len(entry_features) > 0:
                # Use the last feature vector
                feature_vector = entry_features[-1]
                label = 1 if success else 0
                
                self._training_data.append(feature_vector)
                self._training_labels.append(label)
                
                # Trim if too many samples
                if len(self._training_data) > self._max_training_samples:
                    self._training_data = self._training_data[-self._max_training_samples:]
                    self._training_labels = self._training_labels[-self._max_training_samples:]
                
                # Train if enough samples
                if len(self._training_data) >= self.meta_ensemble.config.min_samples_for_training:
                    self._train_meta_model()
                    
        except Exception as e:
            logger.error(f"Failed to collect training data: {e}")
    
    def _train_meta_model(self):
        """Train meta-model on collected data."""
        if not self.meta_ensemble or len(self._training_data) < 100:
            return
        
        try:
            X = np.array(self._training_data)
            y = np.array(self._training_labels)
            
            # Train with validation split
            metrics = self.meta_ensemble.train_with_validation_split(X, y)
            
            logger.info(
                f"Meta-model trained: accuracy={metrics.get('test_accuracy', 0):.3f}, "
                f"auc={metrics.get('test_auc', 0):.3f}"
            )
            
            # Save the trained model
            self.meta_ensemble.save()
            
        except Exception as e:
            logger.error(f"Failed to train meta-model: {e}")
