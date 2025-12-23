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

import sys
from pathlib import Path
# Add project root to sys.path to allow imports of src modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
        """Load trained ML model and feature engineer with robust fallback."""
        try:
            # Find latest model
            model_dir = Path("user_data/models")
            model_files = list(model_dir.glob("*.pkl"))
            
            if not model_files:
                logger.warning("No ML models found in user_data/models")
                self._create_fallback_model()
                return
            
            # Filter out small files (likely not proper models)
            model_files = [f for f in model_files if f.stat().st_size > 10000]
            if not model_files:
                logger.warning("No valid ML models found (files too small)")
                self._create_fallback_model()
                return
            
            # Get latest model by modification time
            latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
            
            logger.info(f"Loading ML model: {latest_model.name}")
            with open(latest_model, 'rb') as f:
                self.ml_model = pickle.load(f)
            
            # Initialize feature engineer (same as used in training)
            if USE_CUSTOM_MODULES:
                from src.ml.training.feature_engineering import FeatureEngineer, FeatureConfig
                # Create config with correlation removal disabled to match model's expected features
                config = FeatureConfig(remove_correlated=False)
                self.feature_engineer = FeatureEngineer(config)
            else:
                self.feature_engineer = None
            
            # Validate loaded model
            if self.ml_model is None:
                logger.warning("Loaded model is None, creating fallback")
                self._create_fallback_model()
            else:
                logger.info(f"ML model loaded successfully. Model type: {type(self.ml_model).__name__}")
                
                # Log model info
                if hasattr(self.ml_model, 'n_estimators'):
                    logger.info(f"Model has {self.ml_model.n_estimators} estimators")
                if hasattr(self.ml_model, 'feature_importances_'):
                    expected_features = len(self.ml_model.feature_importances_)
                    logger.info(f"Model has {expected_features} features")
                    
                    # Check if feature engineer will produce matching features
                    # We'll check this later during prediction, but log warning now
                    logger.warning(f"Model expects {expected_features} features. Feature engineering may produce different count.")
                else:
                    # If model doesn't have feature_importances_, we can't check
                    logger.warning("Model doesn't have feature_importances_ attribute. Cannot verify feature count.")
            
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            logger.info("Creating fallback model...")
            self._create_fallback_model()
    
    def _create_fallback_model(self):
        """Create a simple fallback model when no trained model is available."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            import numpy as np
            
            logger.info("Creating fallback RandomForest model...")
            
            # Create a simple model with balanced class weights
            self.ml_model = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            # Train on dummy data (just to initialize)
            X_dummy = np.random.randn(100, 10)
            y_dummy = np.random.randint(0, 2, 100)
            self.ml_model.fit(X_dummy, y_dummy)
            
            # Initialize feature engineer
            if USE_CUSTOM_MODULES:
                from src.ml.training.feature_engineering import FeatureEngineer
                self.feature_engineer = FeatureEngineer()
            else:
                self.feature_engineer = None
            
            logger.info("Fallback model created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create fallback model: {e}")
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
        """Calculate ML model predictions using improved feature engineering."""
        
        if self.ml_model is None or self.feature_engineer is None:
            # No ML model available
            dataframe['ml_prediction'] = 0.5
            dataframe['ml_confidence'] = 0.5
            dataframe['ml_signal'] = 0
            return dataframe
        
        try:
            # Prepare features for ML model using the same feature engineering as training
            # Create feature DataFrame with required columns
            feature_df = dataframe[['open', 'high', 'low', 'close', 'volume']].copy()
            
            # Use the feature engineer to create the same features as in training
            if self.feature_engineer.is_fitted():
                # If feature engineer is fitted, use transform (for test data)
                feature_df = self.feature_engineer.transform(feature_df)
            else:
                # If not fitted, use fit_transform (this should only happen once)
                feature_df = self.feature_engineer.fit_transform(feature_df)
            
            # Get feature names that the model expects
            feature_names = self.feature_engineer.get_feature_names()
            
            # Ensure we have all required features
            if feature_names:
                # Select only the features that exist in our dataframe
                available_features = [f for f in feature_names if f in feature_df.columns]
                
                if available_features:
                    X = feature_df[available_features].values
                    
                    # Check if feature count matches model's expected features
                    if hasattr(self.ml_model, 'feature_importances_'):
                        expected_features = len(self.ml_model.feature_importances_)
                        if len(available_features) != expected_features:
                            logger.warning(
                                f"Feature count mismatch: model expects {expected_features}, "
                                f"got {len(available_features)}. Using fallback predictions."
                            )
                            # Use fallback predictions
                            dataframe['ml_prediction'] = 0.5
                            dataframe['ml_confidence'] = 0.5
                            dataframe['ml_signal'] = 0
                            return dataframe
                    
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
                        
                        # Calculate prediction quality metrics
                        prediction_mean = dataframe['ml_prediction'].mean()
                        confidence_mean = dataframe['ml_confidence'].mean()
                        
                        logger.debug(
                            f"ML predictions: mean={prediction_mean:.3f}, "
                            f"confidence={confidence_mean:.3f}, "
                            f"features={len(available_features)}"
                        )
                    else:
                        # Get class predictions
                        predictions = self.ml_model.predict(X)
                        dataframe['ml_prediction'] = predictions
                        dataframe['ml_confidence'] = 0.5  # Default confidence
                else:
                    logger.warning("No matching features found for ML model")
                    dataframe['ml_prediction'] = 0.5
                    dataframe['ml_confidence'] = 0.5
            else:
                logger.warning("Feature engineer has no feature names")
                dataframe['ml_prediction'] = 0.5
                dataframe['ml_confidence'] = 0.5
            
            # Create binary signal (1 if prediction > 0.5)
            dataframe['ml_signal'] = (dataframe['ml_prediction'] > 0.5).astype(int)
            
            # Calculate signal statistics
            signal_count = dataframe['ml_signal'].sum()
            signal_pct = signal_count / len(dataframe) * 100 if len(dataframe) > 0 else 0
            
            if signal_count > 0 or len(dataframe) % 100 == 0:
                logger.info(
                    f"ML predictions for {metadata['pair']}: "
                    f"{signal_count} signals ({signal_pct:.1f}%), "
                    f"mean prediction={dataframe['ml_prediction'].mean():.3f}"
                )
            
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
        Define entry conditions with dynamic probability thresholds.
        
        Based on roadmap Phase 4 improvements:
        1. Dynamic probability thresholds based on market regime
        2. Simplified entry conditions (only 2-3 key conditions)
        3. Regime-based position sizing
        4. Adaptive stop loss based on model confidence
        
        Entry Logic:
        - Buy when ML prediction probability > dynamic_threshold
        - Dynamic threshold adjusts based on market regime and model confidence
        - Simplified conditions: ML signal + trend filter only
        """
        
        # Initialize enter_long column
        dataframe['enter_long'] = 0
        
        # Calculate dynamic threshold based on market regime
        dynamic_threshold = self._calculate_dynamic_threshold(dataframe)
        
        # Simplified entry conditions (only 2 conditions per roadmap)
        # 1. ML prediction above dynamic threshold
        # 2. Price above EMA 200 (trend filter)
        entry_condition = (
            (dataframe['ml_prediction'] > dynamic_threshold) &
            (dataframe['close'] > dataframe['ema_200'])
        )
        
        # Apply regime-based position sizing
        if self._regime_mode == 'high_volatility':
            # Reduce position size in high volatility
            entry_condition = entry_condition & (dataframe['ml_confidence'] > 0.7)
            logger.info(f"High volatility regime: requiring confidence > 0.7")
        elif self._regime_mode == 'low_volatility':
            # Can be more aggressive in low volatility
            entry_condition = entry_condition & (dataframe['ml_confidence'] > 0.5)
            logger.info(f"Low volatility regime: requiring confidence > 0.5")
        
        # Apply entry signals
        dataframe.loc[entry_condition, 'enter_long'] = 1
        
        # Calculate position size based on model confidence
        # Always create position_size column (initialize with NaN)
        dataframe['position_size'] = np.nan
        
        if 'enter_long' in dataframe.columns and dataframe['enter_long'].sum() > 0:
            # Use confidence as position size multiplier (0.5 to 1.0) for entry signals
            dataframe.loc[dataframe['enter_long'] == 1, 'position_size'] = (
                dataframe['ml_confidence'].clip(lower=0.5, upper=1.0)
            )
        
        # Log entry signals
        entry_count = dataframe['enter_long'].sum()
        if entry_count > 0:
            avg_confidence = dataframe.loc[dataframe['enter_long'] == 1, 'ml_confidence'].mean()
            avg_threshold = dynamic_threshold[dataframe['enter_long'] == 1].mean()
            logger.info(
                f"📊 {metadata['pair']}: {entry_count} entry signals "
                f"(avg confidence: {avg_confidence:.3f}, "
                f"avg threshold: {avg_threshold:.3f}, "
                f"regime: {self._regime_mode})"
            )
        
        return dataframe
    
    def _calculate_dynamic_threshold(self, dataframe: DataFrame) -> pd.Series:
        """
        Calculate dynamic probability threshold based on:
        1. Market regime (volatility)
        2. Model confidence
        3. Recent prediction distribution
        
        Based on roadmap: Dynamic probability thresholds instead of fixed 0.55
        
        Returns:
            Series of dynamic thresholds for each row
        """
        base_threshold = 0.55
        
        # Adjust based on market regime
        if self._regime_mode == 'high_volatility':
            # Higher threshold in high volatility (more conservative)
            regime_adjustment = 0.05
        elif self._regime_mode == 'low_volatility':
            # Lower threshold in low volatility (more aggressive)
            regime_adjustment = -0.03
        else:
            regime_adjustment = 0.0
        
        # Adjust based on recent prediction volatility
        if 'ml_prediction' in dataframe.columns:
            # Calculate rolling std of predictions
            pred_std = dataframe['ml_prediction'].rolling(20).std().fillna(0)
            # Higher std -> higher threshold (more conservative)
            volatility_adjustment = pred_std * 0.5
        else:
            volatility_adjustment = 0
        
        # Adjust based on model confidence
        if 'ml_confidence' in dataframe.columns:
            # Higher confidence -> can use lower threshold
            confidence_adjustment = -dataframe['ml_confidence'] * 0.1
        else:
            confidence_adjustment = 0
        
        # Calculate dynamic threshold
        dynamic_threshold = base_threshold + regime_adjustment + volatility_adjustment + confidence_adjustment
        
        # Apply bounds: 0.45 to 0.65
        dynamic_threshold = dynamic_threshold.clip(lower=0.45, upper=0.65)
        
        return dynamic_threshold
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: Optional[float], 
                           max_stake: float, leverage: float, entry_tag: Optional[str],
                           side: str, **kwargs) -> float:
        """
        Custom position sizing based on model confidence and market regime.
        
        Based on roadmap: Regime-based position sizing
        
        Args:
            pair: Trading pair
            current_time: Current time
            current_rate: Current rate
            proposed_stake: Proposed stake amount
            min_stake: Minimum stake
            max_stake: Maximum stake
            leverage: Leverage
            entry_tag: Entry tag
            side: Trade side
            
        Returns:
            Adjusted stake amount
        """
        # Base stake from config
        stake = proposed_stake
        
        # Adjust based on market regime
        if self._regime_mode == 'high_volatility':
            # Reduce position size in high volatility
            stake = stake * 0.7
            logger.info(f"High volatility regime: reducing position size to 70%")
        elif self._regime_mode == 'low_volatility':
            # Increase position size in low volatility
            stake = stake * 1.2
            logger.info(f"Low volatility regime: increasing position size to 120%")
        
        # Apply min/max bounds
        if min_stake is not None:
            stake = max(stake, min_stake)
        stake = min(stake, max_stake)
        
        return stake
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, 
                       after_fill: bool = False, **kwargs) -> Optional[float]:
        """
        Adaptive stop loss based on model confidence and market regime.
        
        Based on roadmap: Adaptive stop loss based on model confidence
        - Higher confidence = tighter stop loss (more conviction)
        - Lower confidence = wider stop loss (more room for error)
        - Regime-based adjustment
        
        Args:
            pair: Trading pair
            trade: Trade object
            current_time: Current time
            current_rate: Current rate
            current_profit: Current profit
            after_fill: After fill flag
            **kwargs: Additional arguments
            
        Returns:
            Adjusted stop loss percentage (negative, e.g., -0.05 for 5% stop loss)
        """
        # Base stop loss from strategy
        base_stoploss = self.stoploss  # e.g., -0.05
        
        # Get model confidence from trade metadata if available
        confidence = 0.5  # default
        if hasattr(trade, 'confidence') and trade.confidence is not None:
            confidence = trade.confidence
        elif 'ml_confidence' in trade.metadata:
            confidence = trade.metadata.get('ml_confidence', 0.5)
        
        # Adjust stop loss based on confidence
        # Higher confidence -> tighter stop loss (more conviction)
        # Lower confidence -> wider stop loss (more room for error)
        confidence_adjustment = (0.5 - confidence) * 0.02  # ±1% adjustment
        
        # Adjust based on market regime
        if self._regime_mode == 'high_volatility':
            # Wider stop loss in high volatility
            regime_adjustment = -0.01  # 1% wider
        elif self._regime_mode == 'low_volatility':
            # Tighter stop loss in low volatility
            regime_adjustment = 0.005  # 0.5% tighter
        else:
            regime_adjustment = 0.0
        
        # Calculate adaptive stop loss
        adaptive_stoploss = base_stoploss + confidence_adjustment + regime_adjustment
        
        # Apply bounds: -0.02 to -0.10 (2% to 10% stop loss)
        adaptive_stoploss = max(adaptive_stoploss, -0.10)  # Not more than 10%
        adaptive_stoploss = min(adaptive_stoploss, -0.02)  # Not less than 2%
        
        # Log adjustment
        logger.info(
            f"Adaptive stop loss for {pair}: "
            f"base={base_stoploss:.3%}, "
            f"confidence={confidence:.3f}, "
            f"regime={self._regime_mode}, "
            f"final={adaptive_stoploss:.3%}"
        )
        
        return adaptive_stoploss
    
    # ==========================================================================
    # EXIT LOGIC
    # ==========================================================================
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define exit conditions.
        
        Exit when:
        - RSI overbought (> sell_rsi)
        - ML prediction drops below threshold (0.45)
        - Simple time-based exit (optional)
        """
        
        conditions = []
        
        # Overbought exit
        overbought = (
            (dataframe['rsi'] > self.sell_rsi.value)
        )
        conditions.append(overbought)
        
        # ML prediction turns negative
        if 'ml_prediction' in dataframe.columns:
            ml_exit = (
                (dataframe['ml_prediction'] < 0.45)
            )
            conditions.append(ml_exit)
        
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
            )
        
        return dataframe
