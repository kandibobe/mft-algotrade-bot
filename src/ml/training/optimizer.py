"""
ML Model Optimizer and Trainer
==============================

Unified service for training and optimizing ML models.
Integrates ML hyperparameter optimization with trading strategy parameters.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

from src.config.unified_config import TradingConfig, TrainingConfig
from src.ml.training.feature_engineering import FeatureEngineer, FeatureConfig
from src.ml.training.labeling import TripleBarrierLabeler, TripleBarrierConfig
from src.ml.training.model_trainer import ModelTrainer, TrainingConfig as ModelTrainerConfig
from src.ml.training.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

class ModelOptimizer:
    """
    Unified optimizer for ML models and strategy parameters.
    
    Supports:
    1. Standard training of single models
    2. Joint optimization of ML hyperparameters and strategy parameters
    """

    def __init__(self, config: TradingConfig, models_dir: str = "user_data/models"):
        self.config = config
        self.training_config = config.training
        self.ml_config = config.ml
        self.models_dir = Path(models_dir)
        
        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.labeler = TripleBarrierLabeler(config=TripleBarrierConfig())
        
        # Registry
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry = ModelRegistry(registry_dir=str(self.models_dir / "registry"))

    def train(self, data: pd.DataFrame, pair: str, optimize: bool = False) -> Dict[str, Any]:
        """
        Train a model for a specific pair.
        
        Args:
            data: OHLCV DataFrame
            pair: Trading pair name
            optimize: Whether to run hyperparameter optimization
            
        Returns:
            Dictionary with training results
        """
        if optimize:
            return self.optimize_model(data, pair)
        else:
            return self.train_single_model(data, pair)

    def train_single_model(self, data: pd.DataFrame, pair: str) -> Dict[str, Any]:
        """Train a single model with fixed parameters."""
        logger.info(f"Starting standard training for {pair}")
        
        # 1. Feature Engineering
        logger.info("Generating features...")
        features = self.feature_engineer.prepare_data(data)
        
        # 2. Labeling
        logger.info("Generating labels...")
        labels = self.labeler.label(data)
        
        # 3. Align data
        common_index = features.index.intersection(labels.index)
        X = features.loc[common_index]
        y = labels.loc[common_index]
        
        # Remove NaNs
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 100:
            raise ValueError(f"Insufficient data for {pair}: {len(X)} samples")
            
        # 4. Train/Test Split
        validation_split = self.training_config.validation_split
        train_size = int(len(X) * (1 - validation_split))
        
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        X_test = X.iloc[train_size:]
        y_test = y.iloc[train_size:]
        
        # 5. Scaling/Feature Selection (Fit on Train)
        X_train = self.feature_engineer.fit_scaler_and_selector(X_train)
        X_test = self.feature_engineer.transform_scaler_and_selector(X_test)
        
        # 6. Train Model
        trainer_config = ModelTrainerConfig(
            model_type=self.training_config.model_type,
            optimize_hyperparams=False,
            n_trials=0,
            models_dir="user_data/models"
        )
        
        trainer = ModelTrainer(trainer_config)
        model, metrics = trainer.train(X_train, y_train, X_test, y_test)
        
        # 7. Save Model
        self._save_model(model, pair, metrics, X.columns.tolist())
        
        return {
            "success": True,
            "metrics": metrics,
            "model": model
        }

    def optimize_model(self, data: pd.DataFrame, pair: str) -> Dict[str, Any]:
        """
        Run hyperparameter optimization using Optuna.
        Incorporates logic from tools/ops/optimize_strategy.py
        """
        logger.info(f"Starting optimization for {pair}")
        
        n_trials = self.training_config.hyperopt_trials
        
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            study_name=f"{pair}_optimization"
        )
        
        def objective(trial):
            # 1. Sample Parameters
            ml_params = self._sample_ml_params(trial)
            trading_params = self._sample_trading_params(trial)
            
            # 2. Feature Engineering (dynamic windows could be added here if needed)
            # For efficiency, we use pre-computed features and just select/transform?
            # Or re-compute if window sizes are optimized.
            # optimize_strategy.py optimized window sizes (short_period, medium_period).
            
            try:
                # Dynamic feature config based on trial
                feature_config = FeatureConfig(
                    short_period=ml_params.get('short_period', 14),
                    medium_period=ml_params.get('medium_period', 50),
                    include_time_features=True
                )
                
                # We need to re-instantiate FeatureEngineer for dynamic windows
                engineer = FeatureEngineer(feature_config)
                features = engineer.transform(data.copy())
                
                # Labeling (dynamic labeling params)
                label_config = TripleBarrierConfig(
                    take_profit=trading_params.get('take_profit', 0.02),
                    stop_loss=trading_params.get('stop_loss', 0.01),
                    max_holding_period=trading_params.get('max_holding_period', 24)
                )
                labeler = TripleBarrierLabeler(label_config)
                labels = labeler.label(features)
                
                # Align and Clean
                common_index = features.index.intersection(labels.index)
                X = features.loc[common_index]
                y = labels.loc[common_index]
                
                valid_mask = ~y.isna()
                X = X.loc[valid_mask].dropna()
                y = y.loc[X.index]
                
                if len(X) < 100:
                    return float('-inf')
                
                # Walk-Forward Validation
                return self._walk_forward_validation(X, y, ml_params, trading_params)
                
            except Exception as e:
                # logger.warning(f"Trial failed: {e}")
                return float('-inf')

        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        best_score = study.best_value
        
        logger.info(f"Optimization complete. Best Score: {best_score:.4f}")
        logger.info(f"Best Params: {best_params}")
        
        # Save best parameters
        self._save_optimization_results(pair, best_params, best_score, study.trials)
        
        return {
            "success": True,
            "best_params": best_params,
            "best_score": best_score
        }

    def _sample_ml_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample ML hyperparameters."""
        model_type = trial.suggest_categorical('model_type', ['random_forest', 'xgboost', 'lightgbm'])
        
        params = {
            'model_type': model_type,
            'short_period': trial.suggest_int('short_period', 7, 21),
            'medium_period': trial.suggest_int('medium_period', 30, 100),
        }
        
        if model_type == 'random_forest':
            params.update({
                'n_estimators': trial.suggest_int('rf_n_estimators', 50, 300),
                'max_depth': trial.suggest_int('rf_max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
            })
        elif model_type == 'xgboost':
            params.update({
                'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 300),
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
                'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
            })
        elif model_type == 'lightgbm':
            params.update({
                'n_estimators': trial.suggest_int('lgb_n_estimators', 50, 300),
                'num_leaves': trial.suggest_int('lgb_num_leaves', 20, 150),
                'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.3, log=True),
            })
            
        return params

    def _sample_trading_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample trading/labeling parameters."""
        return {
            'take_profit': trial.suggest_float('take_profit', 0.005, 0.05),
            'stop_loss': trial.suggest_float('stop_loss', 0.005, 0.03),
            'max_holding_period': trial.suggest_int('max_holding_period', 6, 48),
            'entry_threshold': trial.suggest_float('entry_threshold', 0.5, 0.8),
        }

    def _walk_forward_validation(self, X: pd.DataFrame, y: pd.Series, 
                               ml_params: Dict[str, Any], trading_params: Dict[str, Any]) -> float:
        """Perform walk-forward validation maximizing Sharpe Ratio."""
        from sklearn.model_selection import TimeSeriesSplit
        
        # 5-fold Time Series Split
        tscv = TimeSeriesSplit(n_splits=5)
        returns_list = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            model = self._create_model_from_params(ml_params)
            model.fit(X_train, y_train)
            
            # Predict
            probs = model.predict_proba(X_test)
            
            # Simulate Trading
            fold_returns = self._simulate_trading(probs, y_test, trading_params)
            returns_list.extend(fold_returns)
            
        if not returns_list:
            return float('-inf')
            
        # Calculate Sharpe Ratio
        returns = np.array(returns_list)
        if np.std(returns) == 0:
            return float('-inf')
            
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(288 * 365) # Annualized
        return sharpe

    def _create_model_from_params(self, params: Dict[str, Any]):
        """Create model instance."""
        model_type = params.get('model_type', 'lightgbm')
        
        if model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 10),
                min_samples_split=params.get('min_samples_split', 2),
                n_jobs=-1, random_state=42
            )
        elif model_type == 'xgboost':
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.1),
                n_jobs=-1, random_state=42, use_label_encoder=False, eval_metric='logloss'
            )
        elif model_type == 'lightgbm':
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                n_estimators=params.get('n_estimators', 100),
                num_leaves=params.get('num_leaves', 31),
                learning_rate=params.get('learning_rate', 0.1),
                n_jobs=-1, random_state=42, verbose=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _simulate_trading(self, probs: np.ndarray, y_true: pd.Series, params: Dict[str, Any]) -> List[float]:
        """Simulate returns based on predictions and thresholds."""
        threshold = params.get('entry_threshold', 0.6)
        tp = params.get('take_profit', 0.02)
        sl = params.get('stop_loss', 0.01)
        
        returns = []
        for i, prob_arr in enumerate(probs):
            # Assume binary [prob_0, prob_1] or multi-class where last is 1 (Buy)
            buy_prob = prob_arr[-1] if len(prob_arr) > 1 else prob_arr[0]
            true_label = y_true.iloc[i]
            
            if buy_prob > threshold:
                if true_label == 1:
                    returns.append(tp)
                elif true_label == -1: # Assuming -1 is Short/Loss in this simplified view or we map it
                    # If labeler returns -1 for Short, and we only trade Long:
                    # If label is 1 (Price went up), we win TP.
                    # If label is 0 or -1 (Price didn't go up or went down), we hit SL or timeout.
                    # Triple Barrier Labeler usually returns 1 (Top barrier), -1 (Bottom barrier), 0 (Timeout)
                    returns.append(-sl)
                else:
                    # Timeout - exit at close? Or small loss/gain.
                    # Simplified: treat as small friction loss
                    returns.append(-0.001)
        
        return returns

    def _save_model(self, model, pair, metrics, feature_names):
        """Save model and register it."""
        import joblib
        
        # Format: models/{pair}_{model_type}_v{version}_{date}.joblib
        # Example: models/BTC_USDT_xgboost_v1_20240103.joblib
        date_str = datetime.now().strftime('%Y%m%d')
        version = self.training_config.feature_set_version
        model_type = self.training_config.model_type
        pair_slug = pair.replace('/', '_')
        
        model_filename = f"{pair_slug}_{model_type}_{version}_{date_str}"
        model_path = Path("user_data/models") / f"{model_filename}.joblib"
        
        joblib.dump(model, model_path)
            
        self.registry.register_model(
            model_name=pair_slug,
            model_path=str(model_path),
            metrics=metrics,
            training_config=self.training_config.dict(),
            feature_names=feature_names,
            tags=['standard_training', f'version:{version}', f'type:{model_type}']
        )
        logger.info(f"Model saved to {model_path}")

    def _save_optimization_results(self, pair, best_params, best_score, trials):
        """Save optimization results."""
        output_path = Path(f"user_data/models/{pair.replace('/', '_')}_optimization.json")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "pair": pair,
            "best_score": best_score,
            "best_params": best_params,
            "n_trials": len(trials),
            "history": [
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params
                } for t in trials
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Optimization results saved to {output_path}")
