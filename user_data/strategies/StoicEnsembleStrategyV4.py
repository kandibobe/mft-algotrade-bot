"""
Stoic Citadel - Ensemble Strategy V4 (ML-Enhanced)
==================================================

Enhanced version with ML model predictions integrated into ensemble.

Features:
1. Multi-Model ML predictions (Random Forest/XGBoost) per pair
2. Dynamic weighting between ML and traditional signals
3. Confidence scores from ML model probabilities
4. All features from V3 (meta-learning, regime detection, etc.)

Philosophy: "Combine human wisdom with machine intelligence."
Financial Logic:
- We do not trust raw ML probabilities blindly. We rank them against recent history.
- We do not hold losing trades hoping for a miracle. We decay our profit target over time.
- We adapt position sizing and stop losses based on market volatility (Regime Detection).

Author: Stoic Citadel Team
Version: 4.2.1 (Calibrated & Decaying)
"""

import sys
import re
from pathlib import Path
# Add project root to sys.path to allow imports of src modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import pickle
import joblib
import logging
from typing import Optional, Dict, Any, Tuple, List, Union
from datetime import datetime, timedelta
from functools import reduce

import pandas as pd
import numpy as np
from pandas import DataFrame

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter
from freqtrade.persistence import Trade

# Try to import custom modules
try:
    from src.utils.indicators import (
        calculate_ema, calculate_rsi, calculate_macd,
        calculate_atr, calculate_bollinger_bands,
        calculate_stochastic, calculate_adx, calculate_obv
    )
    from src.utils.regime_detection import calculate_regime_score, get_regime_parameters
    from src.ml.training.feature_engineering import FeatureEngineer, FeatureConfig
    from src.ml.calibration import ProbabilityCalibrator
    USE_CUSTOM_MODULES = True
except ImportError as e:
    USE_CUSTOM_MODULES = False
    import talib.abstract as ta
    logger = logging.getLogger(__name__)
    logger.warning(f"Custom modules not available: {e}. Falling back to TA-Lib.")

logger = logging.getLogger(__name__)

# Global cache for ML models to avoid reloading in multiple instances within the same process
_MODEL_CACHE = {
    'models': {},
    'feature_engineers': {}
}

def get_cached_model_and_fe(pair: str, base_dir: Path) -> Tuple[Any, Any]:
    """
    Get model and feature engineer from global cache, loading if necessary.
    This function is outside the class to avoid pickling issues.
    """
    global _MODEL_CACHE
    
    # Normalize pair name
    normalized_pair = pair.replace('/', '_')
    
    # Return if already cached
    if normalized_pair in _MODEL_CACHE['models']:
        return _MODEL_CACHE['models'][normalized_pair], _MODEL_CACHE['feature_engineers'].get(normalized_pair)
        
    # Try to load
    try:
        model_dir = base_dir / "user_data/models"
        if not model_dir.exists():
            # Try absolute path or relative to current dir
            model_dir = Path("user_data/models")
            
        if not model_dir.exists():
            return None, None
            
        # Find latest model for this pair
        model_files = list(model_dir.glob(f"{normalized_pair}_*.pkl"))
        if not model_files:
            return None, None
            
        # Sort by timestamp (latest last)
        # Filename format: PAIR_YYYYMMDD_HHMMSS.pkl
        model_files.sort(key=lambda f: f.stem.split('_')[-2] + f.stem.split('_')[-1] if len(f.stem.split('_')) >= 3 else f.name)
        latest_model_path = model_files[-1]
        
        logger.info(f"Loading ML model for {pair}: {latest_model_path.name}")
        
        # Load model
        try:
            model = joblib.load(latest_model_path, mmap_mode='r')
        except Exception:
            with open(latest_model_path, 'rb') as f:
                model = pickle.load(f)
                
        # Load scaler
        scaler_path = model_dir / f"{latest_model_path.stem}_scaler.joblib"
        feature_engineer = None
        
        if scaler_path.exists() and USE_CUSTOM_MODULES:
            config = FeatureConfig(remove_correlated=True, scale_features=True)
            feature_engineer = FeatureEngineer(config)
            feature_engineer.load_scaler(str(scaler_path))
            
        # Update cache
        _MODEL_CACHE['models'][normalized_pair] = model
        if feature_engineer:
            _MODEL_CACHE['feature_engineers'][normalized_pair] = feature_engineer
            
        return model, feature_engineer
        
    except Exception as e:
        logger.error(f"Error loading model for {pair}: {e}")
        return None, None

class StoicEnsembleStrategyV4(IStrategy):
    """
    ML-enhanced ensemble strategy with trained model predictions.
    
    Combines:
    1. ML model predictions (Random Forest/XGBoost)
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
    
    # Dynamic Entry Parameters (Calibration)
    entry_prob_percentile = IntParameter(80, 99, default=95, space='buy')
    entry_prob_window = IntParameter(500, 3000, default=2000, space='buy')
    
    # Volatility Targeting Parameters
    atr_exit_mult = DecimalParameter(1.5, 3.0, default=2.0, space='sell')
    atr_stop_mult = DecimalParameter(1.0, 2.5, default=1.5, space='sell')
    
    # Dynamic Exit Parameters (Linear Decay)
    exit_decay_start_profit = DecimalParameter(0.01, 0.03, default=0.015, space='sell')
    exit_decay_end_profit = DecimalParameter(0.001, 0.005, default=0.002, space='sell')
    
    sell_rsi = IntParameter(65, 85, default=75, space="sell")
    sell_hold_time = IntParameter(12, 48, default=24, space="sell")  # candles
    
    # Exit optimization parameters
    opt_stoploss = DecimalParameter(-0.10, -0.02, default=-0.05, space='sell')
    
    # Timeout Exit (New Feature)
    max_hold_duration = IntParameter(200, 1440, default=1440, space='sell')

    # ROI table parameters for hyperopt
    roi_p1 = DecimalParameter(0.05, 0.20, default=0.12, space="roi")
    roi_p2 = DecimalParameter(0.02, 0.10, default=0.06, space="roi")
    roi_p3 = DecimalParameter(0.01, 0.05, default=0.02, space="roi")
    roi_t1 = IntParameter(10, 60, default=30, space="roi")
    roi_t2 = IntParameter(30, 120, default=60, space="roi")
    roi_t3 = IntParameter(60, 240, default=120, space="roi")
    
    # Trailing stop parameters
    trailing_stop = CategoricalParameter([True, False], default=False, space="trailing")
    trailing_stop_positive = DecimalParameter(0.005, 0.05, default=0.005, space="trailing")
    trailing_stop_positive_offset = DecimalParameter(0.01, 0.10, default=0.01, space="trailing")
    trailing_only_offset_is_reached = True
    
    # Fixed exit parameters
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    
    @property
    def exit_profit_only(self):
        if hasattr(self, '_custom_exit_profit_only') and self._custom_exit_profit_only is not None:
            return self._custom_exit_profit_only
        return False
    
    @exit_profit_only.setter
    def exit_profit_only(self, value):
        self._custom_exit_profit_only = value
    
    @property
    def ignore_roi_if_entry_signal(self):
        if hasattr(self, '_custom_ignore_roi_if_entry_signal') and self._custom_ignore_roi_if_entry_signal is not None:
            return self._custom_ignore_roi_if_entry_signal
        return False
    
    @ignore_roi_if_entry_signal.setter
    def ignore_roi_if_entry_signal(self, value):
        self._custom_ignore_roi_if_entry_signal = value
    
    # ROI
    minimal_roi = {
        "0": 0.12,
        "30": 0.06,
        "60": 0.04,
        "120": 0.02,
        "240": 0.01
    }
    
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
    
    @property
    def minimal_roi(self):
        if hasattr(self, '_custom_minimal_roi') and self._custom_minimal_roi is not None:
            # Check if it's a valid ROI table (all keys must be numeric strings)
            # This prevents loading "parameter dumps" from JSON as ROI tables
            try:
                all(int(k) >= 0 for k in self._custom_minimal_roi.keys())
                return self._custom_minimal_roi
            except ValueError:
                pass
        
        # Dynamic ROI from Hyperopt Parameters
        return {
            "0": self.roi_p1.value + self.roi_p2.value + self.roi_p3.value,
            str(self.roi_t1.value): self.roi_p2.value + self.roi_p3.value,
            str(self.roi_t1.value + self.roi_t2.value): self.roi_p3.value,
            str(self.roi_t1.value + self.roi_t2.value + self.roi_t3.value): 0,
            "120": 0.01  # Constraint: Force exit if profit > 1% after 120 minutes
        }
    
    @minimal_roi.setter
    def minimal_roi(self, value):
        self._custom_minimal_roi = value
    
    @property
    def stoploss(self):
        if hasattr(self, '_custom_stoploss') and self._custom_stoploss is not None:
            return self._custom_stoploss
        return self.opt_stoploss.value

    @stoploss.setter
    def stoploss(self, value):
        # Only set custom stoploss if it explicitly overrides the parameter
        # This prevents _normalize_attributes from locking the default value
        if value is not None:
            val_float = float(value)
            # Use a small epsilon for float comparison
            if abs(val_float - float(self.opt_stoploss.value)) > 0.000001:
                self._custom_stoploss = val_float
    
    @property
    def trailing_stop(self):
        if hasattr(self, '_custom_trailing_stop') and self._custom_trailing_stop is not None:
            return self._custom_trailing_stop
        if hasattr(self.__class__, 'trailing_stop'):
            param = getattr(self.__class__, 'trailing_stop')
            if hasattr(param, 'value'):
                return param.value
        return True
    
    @trailing_stop.setter
    def trailing_stop(self, value):
        self._custom_trailing_stop = value
    
    @property
    def trailing_stop_positive(self):
        if hasattr(self, '_custom_trailing_stop_positive') and self._custom_trailing_stop_positive is not None:
            return self._custom_trailing_stop_positive
        if hasattr(self.__class__, 'trailing_stop_positive'):
            param = getattr(self.__class__, 'trailing_stop_positive')
            if hasattr(param, 'value'):
                return param.value
        return 0.005
    
    @trailing_stop_positive.setter
    def trailing_stop_positive(self, value):
        self._custom_trailing_stop_positive = value
    
    @property
    def trailing_stop_positive_offset(self):
        if hasattr(self, '_custom_trailing_stop_positive_offset') and self._custom_trailing_stop_positive_offset is not None:
            return self._custom_trailing_stop_positive_offset
        if hasattr(self.__class__, 'trailing_stop_positive_offset'):
            param = getattr(self.__class__, 'trailing_stop_positive_offset')
            if hasattr(param, 'value'):
                return param.value
        return 0.01
    
    @trailing_stop_positive_offset.setter
    def trailing_stop_positive_offset(self, value):
        self._custom_trailing_stop_positive_offset = value
    
    
    # ==========================================================================
    # INTERNAL STATE
    # ==========================================================================
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._regime_mode = 'normal'
        self._regime_params = {}
        self._last_regime_update = None
        
        self.best_params = None
        
        # Custom parameters for hyperopt support
        self._custom_minimal_roi = None
        self._custom_stoploss = None
        self._custom_trailing_stop = None
        self._custom_trailing_stop_positive = None
        self._custom_trailing_stop_positive_offset = None
        self._custom_exit_profit_only = None
        self._custom_ignore_roi_if_entry_signal = None
        
        # Load best hyperparameters from config
        self._load_best_params()
        
        # NOTE: We do NOT load ML models here to avoid pickling them to workers.
        # They are loaded lazily via get_cached_model_and_fe
        
        # Initialize Calibrator
        self.calibrator = None
        if USE_CUSTOM_MODULES:
            try:
                # Top 95% (0.95) threshold over last 2000 candles
                self.calibrator = ProbabilityCalibrator(window_size=2000, percentile_threshold=0.95)
                logger.info("ProbabilityCalibrator initialized: Top 5% logic active.")
            except Exception as e:
                logger.warning(f"Failed to initialize ProbabilityCalibrator: {e}")
        
        logger.info("StoicEnsembleStrategyV4 initialized with Multi-Model integration")
    
    def _load_best_params(self) -> None:
        """Load best hyperparameters from user_data/model_best_params.json."""
        try:
            params_path = Path("user_data/model_best_params.json")
            if params_path.exists():
                with open(params_path, 'r') as f:
                    self.best_params = json.load(f)
                logger.info(f"Loaded optimized params: {self.best_params}")
            else:
                logger.warning(f"Best parameters file not found at {params_path}")
                self.best_params = None
        except Exception as e:
            logger.error(f"Failed to load best parameters: {e}")
            self.best_params = None
    
    def _get_pair_model_and_fe(self, pair: str) -> Tuple[Any, Any]:
        """Get model and feature engineer for a given pair from global cache."""
        # This uses the global function which handles lazy loading
        # We pass the project root (assuming we are running from project root)
        return get_cached_model_and_fe(pair, Path("."))
    
    # ==========================================================================
    # INDICATORS
    # ==========================================================================
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate all technical indicators and ML predictions.
        
        Wraps calculation in try/except to prevent strategy crash on single pair failure.
        """
        try:
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
                
        except Exception as e:
            logger.error(f"Error in populate_indicators for {metadata.get('pair')}: {e}")
            # Ensure critical columns exist even on failure to prevent downstream crashes
            if 'ml_prediction' not in dataframe.columns:
                dataframe['ml_prediction'] = 0.5
            if 'ml_confidence' not in dataframe.columns:
                dataframe['ml_confidence'] = 0.5
            if 'ml_signal' not in dataframe.columns:
                dataframe['ml_signal'] = 0
            if 'ensemble_score' not in dataframe.columns:
                dataframe['ensemble_score'] = 0.5
        
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
        """Calculate ML model predictions with robust error handling."""
        
        pair = metadata['pair']
        ml_model, feature_engineer = self._get_pair_model_and_fe(pair)
        
        if ml_model is None or feature_engineer is None:
            # No ML model available for this pair
            if not getattr(self, '_logged_missing_model_' + pair, False):
                logger.warning(f"No ML model or feature engineer found for {pair}. Using neutral prediction.")
                setattr(self, '_logged_missing_model_' + pair, True)
                
            dataframe['ml_prediction'] = 0.5
            dataframe['ml_confidence'] = 0.5
            dataframe['ml_signal'] = 0
            return dataframe
        
        try:
            # Transform features using the same pipeline as training
            if not feature_engineer.is_fitted():
                logger.warning(f"Feature engineer scaler for {pair} is not fitted! Predictions may be inaccurate.")
            
            # Prepare dataframe for FeatureEngineer (needs DatetimeIndex)
            df_for_fe = dataframe.copy()
            if 'date' in df_for_fe.columns and not isinstance(df_for_fe.index, pd.DatetimeIndex):
                df_for_fe.set_index('date', inplace=True)
            
            # Transform data (generates features and applies scaling)
            X_df = feature_engineer.transform(df_for_fe)
            
            # Align features with model expectations
            if hasattr(ml_model, 'feature_names_in_'):
                expected_features = list(ml_model.feature_names_in_)
                
                # Check for missing features
                missing_features = [f for f in expected_features if f not in X_df.columns]
                if missing_features:
                    # Log once
                    if not getattr(self, '_logged_missing_features_' + pair, False):
                        logger.warning(f"Missing features for {pair}: {missing_features[:5]}... Total: {len(missing_features)}")
                        setattr(self, '_logged_missing_features_' + pair, True)
                        
                    # Fill missing with 0 to prevent crash (suboptimal but robust)
                    for f in missing_features:
                        X_df[f] = 0
                
                # Reorder columns to match model
                X_df = X_df[expected_features]
            
            # Make predictions
            if hasattr(ml_model, 'predict_proba'):
                # Get probability predictions
                proba = ml_model.predict_proba(X_df)
                
                # Align prediction index with original dataframe
                predictions = pd.Series(index=dataframe.index, dtype=float)
                confidence = pd.Series(index=dataframe.index, dtype=float)
                
                # Map X_df index (DatetimeIndex) back to dataframe index (RangeIndex)
                if isinstance(X_df.index, pd.DatetimeIndex) and not isinstance(dataframe.index, pd.DatetimeIndex):
                    if 'date' in dataframe.columns:
                        # Match by date
                        temp_df = pd.DataFrame(index=X_df.index)
                        if proba.shape[1] > 1:
                            if hasattr(ml_model, 'classes_'):
                                classes = list(ml_model.classes_)
                                if 1 in classes:
                                    col_idx = classes.index(1)
                                    temp_df['pred'] = proba[:, col_idx]
                                else:
                                    temp_df['pred'] = proba[:, -1]
                            else:
                                temp_df['pred'] = proba[:, -1]
                        else:
                            temp_df['pred'] = proba[:, 0]
                        
                        temp_df['conf'] = np.max(proba, axis=1)
                        
                        # Merge back
                        merged = dataframe[['date']].reset_index().merge(temp_df, left_on='date', right_index=True, how='left')
                        merged.set_index('index', inplace=True)
                        
                        predictions = merged['pred']
                        confidence = merged['conf']
                    else:
                        logger.warning(f"Dataframe for {pair} missing 'date' column, cannot align predictions")
                else:
                    # Indices match type
                    if proba.shape[1] > 1:
                        predictions.loc[X_df.index] = proba[:, 1]
                    else:
                        predictions.loc[X_df.index] = proba[:, 0]
                    confidence.loc[X_df.index] = np.max(proba, axis=1)
                
                # Fill missing values (initial rows) with neutral
                dataframe['ml_prediction'] = predictions.fillna(0.5)
                dataframe['ml_confidence'] = confidence.fillna(0.5)
            else:
                # Class predictions
                preds = ml_model.predict(X_df)
                
                predictions = pd.Series(index=dataframe.index, dtype=int)
                
                if isinstance(X_df.index, pd.DatetimeIndex) and not isinstance(dataframe.index, pd.DatetimeIndex):
                    if 'date' in dataframe.columns:
                        temp_df = pd.DataFrame({'pred': preds}, index=X_df.index)
                        merged = dataframe[['date']].reset_index().merge(temp_df, left_on='date', right_index=True, how='left')
                        merged.set_index('index', inplace=True)
                        predictions = merged['pred']
                else:
                    predictions.loc[X_df.index] = preds
                
                dataframe['ml_prediction'] = predictions.fillna(0)
                dataframe['ml_confidence'] = 0.5
            
            # Create binary signal (1 if prediction > 0.5)
            dataframe['ml_signal'] = (dataframe['ml_prediction'] > 0.5).astype(int)
            
        except Exception as e:
            if not getattr(self, '_logged_error_' + pair, False):
                logger.error(f"ML prediction failed for {pair}: {e}")
                setattr(self, '_logged_error_' + pair, True)
            
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
            
        except Exception as e:
            self._regime_mode = 'normal'
    
    # ==========================================================================
    # ENTRY LOGIC
    # ==========================================================================
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry conditions using dynamic probability calibration.
        
        Financial Logic:
        1. We calculate the relative rank of the current ML prediction (Percentile).
        2. We require the prediction to be in the Top 10% of recent history (0.90).
        3. We also require a minimum absolute floor (0.35) to filter out garbage.
        """
        dataframe['enter_long'] = 0
        
        if 'ml_prediction' not in dataframe.columns:
            return dataframe

        # Use Calibrator if available
        if self.calibrator:
            # Calibrate probabilities (Top X% logic)
            # We use Hyperopt parameters to tune the strictness
            window = self.entry_prob_window.value
            percentile = self.entry_prob_percentile.value / 100.0
            
            is_calibrated_signal = self.calibrator.is_signal(
                dataframe['ml_prediction'], 
                window_size=window, 
                threshold=percentile
            )
            
            # Combine with raw floor (Safety Net)
            # Even if it's the best prediction in weeks, if it's < 0.35, it's too weak.
            min_probability_floor = 0.35
            if self._regime_mode == 'high_volatility':
                min_probability_floor = 0.40  # Be stricter in chaos
            
            # Volatility Filter: ATR > 0.5% of price OR ADX > 25 (Strong trend)
            # This filters out dead markets where fees > volatility
            volatility_filter = (
                (dataframe['atr'] > dataframe['close'] * 0.005) |
                (dataframe['adx'] > 25)
            )

            entry_condition = (
                is_calibrated_signal &
                (dataframe['ml_prediction'] > min_probability_floor) &
                (dataframe['volume'] > 0) &
                volatility_filter
            )
        else:
            # Fallback to absolute threshold (Legacy)
            entry_condition = (
                (dataframe['ml_prediction'] > 0.55) &
                (dataframe['volume'] > 0)
            )
        
        # Apply entry signals
        dataframe.loc[entry_condition, 'enter_long'] = 1
        
        return dataframe

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: Optional[float], 
                           max_stake: float, leverage: float, entry_tag: Optional[str],
                           side: str, **kwargs) -> float:
        """Custom position sizing based on market regime."""
        stake = proposed_stake
        
        # Adjust based on market regime
        if self._regime_mode == 'high_volatility':
            # Reduce position size in high volatility
            stake = stake * 0.7
        elif self._regime_mode == 'low_volatility':
            # Increase position size in low volatility
            stake = stake * 1.2
        
        if min_stake is not None:
            stake = max(stake, min_stake)
        stake = min(stake, max_stake)
        
        return stake
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, 
                       after_fill: bool = False, **kwargs) -> Optional[float]:
        """
        Dynamic Stop Loss based on ATR.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Stop Loss = Entry - (Multiplier * ATR)
        # Calculate as percentage relative to current price (Freqtrade requirement)
        # Actually Freqtrade expects stoploss as ratio from OPEN price.
        # But custom_stoploss return value is absolute price OR ratio relative to open?
        # Docs: "return value should be the stoploss percentage (e.g. -0.05 for -5%)"
        # The percentage is usually relative to the current rate if calculating trailing, 
        # but here we want a fixed distance from entry?
        # Freqtrade logic: If returning a value, it updates the stoploss.
        # Ideally, we want a fixed stop loss distance calculated at entry and kept?
        # Or dynamic? Dynamic based on current ATR is fine.
        
        # Let's use current ATR to set the stop distance.
        atr = last_candle['atr']
        stop_distance = atr * float(self.atr_stop_mult.value)
        
        # Calculate percentage: stop_distance / current_rate
        # We return negative.
        stop_loss_pct = stop_distance / current_rate
        
        return -stop_loss_pct
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define exit conditions."""
        conditions = []
        
        # Overbought exit
        overbought = (dataframe['rsi'] > self.sell_rsi.value)
        conditions.append(overbought)
        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'exit_long'
            ] = 1
        
        return dataframe
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        """
        Volatility-Based Profit Target & Timeout Exit.
        """
        # Timeout Exit: Close trade if held longer than max_hold_duration
        if (current_time - trade.open_date_utc).total_seconds() / 60 > self.max_hold_duration.value:
            return 'timeout_decay'

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Volatility Target: 2.0 * ATR
        atr = last_candle['atr']
        target_profit_abs = atr * float(self.atr_exit_mult.value)
        
        # Convert to percentage
        target_profit_pct = target_profit_abs / current_rate
        
        if current_profit > target_profit_pct:
            return "volatility_profit"
            
        return None

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        # Allow StopLoss and ROI if profit is real
        if sell_reason == 'roi':
            current_profit = trade.calc_profit_ratio(rate)
            # If profit is less than 0.3% (0.003), REJECT the exit
            if current_profit < 0.003:
                return False
        return True
