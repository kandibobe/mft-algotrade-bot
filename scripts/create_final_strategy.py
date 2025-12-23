import sys
from pathlib import Path

# Original content from git show (copied from earlier read)
original_content = '''"""
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
                from src.ml.training.feature_engineering import FeatureEngineer
                self.feature_engineer = FeatureEngineer()
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
                    logger.info(f"Model has {len(self.ml_model.feature_importances_)} features")
            
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
                    dataframe['ml_prediction
