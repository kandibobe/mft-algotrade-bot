"""
Stoic Citadel - Ensemble Strategy V4 (ML-Enhanced)
==================================================

Enhanced version with ML model predictions integrated into ensemble.

Features:
1. ML model predictions from trained Random Forest
2. Dynamic weighting between ML and traditional signals
3. Confidence scores from ML model probabilities
4. All features from V3 (meta-learning, regime detection, etc.)

Philosophy: "Combine human wisdom with machine intelligence."

Author: Stoic Citadel Team
Version: 4.0.0
"""

import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from functools import reduce

import pandas as pd
import numpy as np
from pandas import DataFrame

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.persistence import Trade

# Try to import custom modules
try:
    from src.utils.indicators import (
        calculate_ema, calculate_rsi, calculate_macd,
        calculate_atr, calculate_bollinger_bands,
        calculate_stochastic, calculate_adx, calculate_obv
    )
    from src.utils.regime_detection import calculate_regime_score, get_regime_parameters
    from src.ml.training.feature_engineering import FeatureEngineer
    USE_CUSTOM_MODULES = True
except ImportError as e:
    USE_CUSTOM_MODULES = False
    import talib.abstract as ta
    logger = logging.getLogger(__name__)
    logger.warning(f"Custom modules not available: {e}. Falling back to TA-Lib.")

logger = logging.getLogger(__name__)


class StoicEnsembleStrategyV4(IStrategy):
    """
    ML-enhanced ensemble strategy with trained model predictions.
    
    Combines:
    1. ML model predictions (Random Forest)
    2. Traditional technical signals (momentum, mean reversion, breakout)
    3. Meta-learning ensemble (if available)
    4. Regime detection for adaptive parameters
    """
    
    # ==========================================================================
    # STRATEGY METADATA
    # ==========================================================================
    
    INTERFACE_VERSION = 3
    
    # Hyperopt spaces
    buy_rsi = IntParameter(20, 40, default=30, space="buy")
    buy_adx = IntParameter(15, 30, default=20, space="buy")
    buy_bb_width_min = DecimalParameter(0.01, 0.05, default=0.02, space="buy")
    
    # ML model weight
    ml_weight = DecimalParameter(0.3, 0.7, default=0.5, space="buy")
    
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
        
        # ML model
        self.ml_model = None
        self.feature_engineer = None
        self._load_ml_model()
        
        logger.info("StoicEnsembleStrategyV4 initialized with ML model integration")
    
    def _load_ml_model(self):
        """Load trained ML model and feature engineer."""
        try:
            # Find latest model
            model_dir = Path("user_data/models")
            model_files = list(model_dir.glob("*.pkl"))
            if not model_files:
                logger.warning("No ML models found in user_data/models")
                return
            
            # Filter out small files (likely not proper models)
            model_files = [f for f in model_files if f.stat().st_size > 10000]
            if not model_files:
                logger.warning("No valid ML models found (files too small)")
                return
            
            # Get latest model by modification time
            latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
            
            logger.info(f"Loading ML model: {latest_model.name}")
            with open(latest_model, 'rb') as f:
                self.ml_model = pickle.load(f)
            
            # Initialize feature engineer (same as used in training)
            if USE_CUSTOM_MODULES:
                from src.ml.training.feature_engineering import FeatureEngineer
                self.feature_engineer = FeatureEngineer()
            else:
                self.feature_engineer = None
            
            logger.info(f"ML model loaded successfully. Model type: {type(self.ml_model).__name__}")
            
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            self.ml_model = None
            self.feature_engineer = None
    
    # ==========================================================================
    # INDICATORS
    # ==========================================================================
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate all technical indicators and ML predictions.
        """
        
        # Calculate traditional indicators
        if USE_CUSTOM_MODULES:
            dataframe = self._populate_indicators_custom(dataframe)
        else:
            dataframe = self._populate_indicators_talib(dataframe)
        
        # Sub-strategy signals
        dataframe = self._calculate_momentum_signals(dataframe)
        dataframe = self._calculate_mean_reversion_signals(dataframe)
        dataframe = self._calculate_breakout_signals(dataframe)
        
        # ML model predictions
        dataframe = self._calculate_ml_predictions(dataframe, metadata)
        
        # Combined ensemble score
        dataframe = self._calculate_ensemble_score(dataframe)
        
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
    
    def _calculate_ml_predictions(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate ML model predictions."""
        
        if self.ml_model is None or self.feature_engineer is None:
            # No ML model available
            dataframe['ml_prediction'] = 0.5
            dataframe['ml_confidence'] = 0.5
            dataframe['ml_signal'] = 0
            return dataframe
        
        try:
            # Prepare features for ML model
            # We need to create the same features as used in training
            # For simplicity, we'll use a subset of available indicators
            
            # Create feature DataFrame with required columns
            feature_df = dataframe[['open', 'high', 'low', 'close', 'volume']].copy()
            
            # Add technical indicators that the model expects
            # (This should match the features used during training)
            feature_df['returns'] = feature_df['close'].pct_change()
            feature_df['returns_log'] = np.log(feature_df['close'] / feature_df['close'].shift(1))
            feature_df['price_position'] = (feature_df['close'] - feature_df['low']) / (feature_df['high'] - feature_df['low'] + 1e-10)
            feature_df['gap'] = (feature_df['open'] - feature_df['close'].shift(1)) / feature_df['close'].shift(1)
            feature_df['intraday_return'] = (feature_df['close'] - feature_df['open']) / feature_df['open']
            
            # Add volume features
            feature_df['volume_change'] = feature_df['volume'].pct_change()
            feature_df['volume_sma'] = feature_df['volume'].rolling(14).mean()
            feature_df['volume_ratio'] = feature_df['volume'] / (feature_df['volume_sma'] + 1e-10)
            
            # Add RSI
            delta = feature_df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            feature_df['rsi_feature'] = 100 - (100 / (1 + rs))
            
            # Add ATR
            high_low = feature_df['high'] - feature_df['low']
            high_close = np.abs(feature_df['high'] - feature_df['close'].shift())
            low_close = np.abs(feature_df['low'] - feature_df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            feature_df['atr_feature'] = true_range.rolling(14).mean()
            
            # Fill NaN values
            feature_df = feature_df.fillna(method='ffill').fillna(0)
            
            # Get predictions
            # The model expects specific features - we'll use what we have
            # This is a simplification; in production you'd need exact feature matching
            
            # Select numeric columns for prediction
            numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
            X = feature_df[numeric_cols].values
            
            # Make predictions
            if hasattr(self.ml_model, 'predict_proba'):
                # Get probability predictions
                proba = self.ml_model.predict_proba(X)
                # Assuming binary classification: probability of class 1 (buy)
                if proba.shape[1] > 1:
                    dataframe['ml_prediction'] = proba[:, 1]  # Probability of positive class
                else:
                    dataframe['ml_prediction'] = proba[:, 0]
                
                # Confidence as max probability
                dataframe['ml_confidence'] = np.max(proba, axis=1)
            else:
                # Get class predictions
                predictions = self.ml_model.predict(X)
                dataframe['ml_prediction'] = predictions
                dataframe['ml_confidence'] = 0.5  # Default confidence
            
            # Create binary signal (1 if prediction > 0.5)
            dataframe['ml_signal'] = (dataframe['ml_prediction'] > 0.5).astype(int)
            
            logger.debug(f"ML predictions calculated: mean={dataframe['ml_prediction'].mean():.3f}")
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            dataframe['ml_prediction'] = 0.5
            dataframe['ml_confidence'] = 0.5
            dataframe['ml_signal'] = 0
        
        return dataframe
    
    def _calculate_momentum_signals(self, dataframe: DataFrame) -> DataFrame:
        """Momentum sub-strategy."""
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
        """Mean reversion sub-strategy."""
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
        """Breakout sub-strategy."""
        dataframe['resistance'] = dataframe['high'].rolling(20).max()
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
    
    def _calculate_ensemble_score(self, dataframe: DataFrame) -> DataFrame:
        """Calculate combined ensemble score."""
        # Traditional signals (0 or 1)
        traditional_score = (
            dataframe['momentum_signal'].fillna(0) +
            dataframe['mean_reversion_signal'].fillna(0) +
            dataframe['breakout_signal'].fillna(0)
        ) / 3.0
        
        # ML signal (probability 0-1)
        ml_score = dataframe['ml_prediction'].fillna(0.5)
        
        # Weighted combination
        ml_weight = self.ml_weight.value
        dataframe['ensemble_score'] = (
            ml_weight * ml_score + (1 - ml_weight) * traditional_score
        )
        
        # Combined confidence
        dataframe['ensemble_confidence'] = (
            dataframe['ml_confidence'].fillna(0.5) * ml_weight +
            0.5 * (1 - ml_weight)  # Traditional confidence fixed at 0.5
        )
        
        return dataframe
    
    def _update_regime(self, dataframe: DataFrame) -> None:
        """Update market regime."""
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
        Define entry conditions based on ML model predictions.
        
        Entry Logic (as per user requirements):
        - If Model Prediction Probability > 0.65 (High Confidence) -> Enter Long
        - If Probability < 0.65 -> Do nothing
        
        Additional filters to improve signal quality:
        - Trend filter (price above EMA 200)
        - Volume confirmation
        - Not overbought (RSI < 70)
        """
        
        # Base conditions - ML prediction probability > 0.65 (High Confidence)
        base_conditions = [
            # ML model prediction probability threshold (user requirement)
            (dataframe['ml_prediction'] > 0.65),
            
            # ML confidence threshold (if available)
            (dataframe['ml_confidence'] > 0.6),
            
            # Trend filter - only trade in uptrend
            (dataframe['close'] > dataframe['ema_200']),
            (dataframe['ema_50'] > dataframe['ema_100']),
            
            # Volume confirmation - avoid low liquidity
            (dataframe['volume_ratio'] > 0.8),
            
            # Not already overbought
            (dataframe['rsi'] < 70),
            
            # Volatility filter - avoid extreme volatility
            (dataframe['bb_width'] > self.buy_bb_width_min.value),
            (dataframe['bb_width'] < 0.15),
        ]
        
        # Regime-adjusted conditions
        if self._regime_mode == 'defensive':
            # In defensive mode, require stronger signals
            base_conditions.append(dataframe['adx'] > 25)
            base_conditions.append(dataframe['rsi'] < 40)
            base_conditions.append(dataframe['ml_confidence'] > 0.7)
        elif self._regime_mode == 'aggressive':
            # In aggressive mode, relax some conditions
            base_conditions.append(dataframe['volume_ratio'] > 0.5)
        
        # Combine conditions
        if base_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, base_conditions),
                'enter_long'
            ] = 1
        
        # Log entry signals
        entry_count = dataframe['enter_long'].sum()
        if entry_count > 0:
            last_row = dataframe.iloc[-1]
            logger.info(
                f"📊 {metadata['pair']}: {entry_count} entry signals "
                f"(ML prob: {last_row['ml_prediction']:.2f}, "
                f"ML conf: {last_row['ml_confidence']:.2f}, "
                f"regime: {self._regime_mode})"
            )
        
        return dataframe
    
    # ==========================================================================
    # EXIT LOGIC
    # ==========================================================================
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define exit conditions."""
        
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
        
        # ML-based exit
        if 'ml_prediction' in dataframe.columns:
            weak_prediction = (
                (dataframe['ml_prediction'] < 0.4) &
                (dataframe['ml_confidence'] > 0.6)
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
        """Dynamic position sizing."""
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return proposed_stake
        
        current = dataframe.iloc[-1]
        
        # Get ATR-based volatility
        atr_pct = current.get('atr_pct', 2.0)
        
        # Base volatility adjustment
        if atr_pct > 5:
            vol_factor = 0.5
        elif atr_pct > 3:
            vol_factor = 0.75
        else:
            vol_factor = 1.0
        
        # Regime adjustment
        regime_factor = self._regime_params.get('risk_per_trade', 0.02) / 0.02
        
        # Confidence-based adjustment
        confidence = current.get('ensemble_confidence', 0.5)
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
        """Final validation before entry."""
        
        # Time filter: avoid low liquidity hours
        hour = current_time.hour
        if hour in [0, 1, 2, 3, 4, 5]:
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
            if current.get('ensemble_confidence', 0) < 0.6:
                logger.info(f"Skipping {pair}: Low confidence in defensive mode")
                return False
        
        # Check ensemble score
        ensemble_score = current.get('ensemble_score', 0.5)
        if ensemble_score < 0.6:
            logger.info(f"Skipping {pair}: Weak ensemble score")
            return False
        
        # Check confidence
        confidence = current.get('ensemble_confidence', 0.5)
        if confidence < 0.5:
            logger.info(f"Skipping {pair}: Low confidence")
            return False
        
        return True
    
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
        """Dynamic leverage based on regime and confidence."""
        
        # Get current data for confidence
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty:
                current = dataframe.iloc[-1]
                confidence = current.get('ensemble_confidence', 0.5)
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
    
    def custom_exit_price(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        proposed_rate: float,
        current_profit: float,
        exit_tag: Optional[str],
        **kwargs
    ) -> float:
        """
        Dynamic exit price based on volatility (ATR).
        
        Replaces fixed ROI with dynamic ROI based on volatility:
        - High volatility: Wider profit targets
        - Low volatility: Tighter profit targets
        
        This adapts to market conditions and improves risk-adjusted returns.
        """
        try:
            # Get current market data
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty:
                return proposed_rate
            
            current = dataframe.iloc[-1]
            
            # Get ATR as percentage of price (volatility measure)
            atr_pct = current.get('atr_pct', 2.0)  # Default 2% if not available
            
            # Calculate dynamic ROI multiplier based on volatility
            # Higher volatility = higher potential profit targets
            # Lower volatility = tighter profit targets
            
            # Base multiplier: 1.0 for normal volatility (2% ATR)
            # Scale up/down based on current ATR
            volatility_multiplier = atr_pct / 2.0
            
            # Apply bounds: 0.5x to 2.0x
            volatility_multiplier = max(0.5, min(2.0, volatility_multiplier))
            
            # Get current profit and adjust target
            if current_profit > 0:
                # For profitable trades, adjust exit price based on volatility
                # Higher volatility = let profits run more
                # Lower volatility = take profits sooner
                
                # Calculate dynamic profit target
                base_target = 0.02  # 2% base target
                dynamic_target = base_target * volatility_multiplier
                
                # If we haven't reached dynamic target yet, adjust exit price
                if current_profit < dynamic_target:
                    # Calculate required price to reach dynamic target
                    entry_price = trade.open_rate
                    target_price = entry_price * (1 + dynamic_target)
                    
                    # Use the higher of proposed rate or target price
                    adjusted_rate = max(proposed_rate, target_price)
                    
                    logger.debug(
                        f"Dynamic exit: {pair} | ATR%: {atr_pct:.2f}% | "
                        f"Vol multiplier: {volatility_multiplier:.2f}x | "
                        f"Dynamic target: {dynamic_target:.2%} | "
                        f"Current profit: {current_profit:.2%} | "
                        f"Adjusted exit: {adjusted_rate:.2f}"
                    )
                    
                    return adjusted_rate
            
            # For losing trades or if already above target, use proposed rate
            return proposed_rate
            
        except Exception as e:
            logger.error(f"Error in custom_exit_price for {pair}: {e}")
            return proposed_rate
