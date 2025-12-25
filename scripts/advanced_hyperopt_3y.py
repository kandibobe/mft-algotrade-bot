#!/usr/bin/env python3
"""
Advanced Hyperparameter Optimization for XGBoost Trading Models - 3 YEAR VERSION
================================================================

Modified version for 3-year retraining as part of Production Hardening.
Saves results to user_data/model_best_params_3y.json

Author: Stoic Citadel Team
Date: December 23, 2025
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import optuna
from optuna.trial import Trial
from optuna.pruners import MedianPruner
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import get_ohlcv
from src.ml.training.feature_engineering import FeatureEngineer
from src.ml.training.labeling import TripleBarrierLabeler, TripleBarrierConfig
from src.utils.risk import calculate_sharpe_ratio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SharpeRatioObjective:
    """
    Objective function for Optuna that optimizes Sharpe Ratio.
    
    Uses binary classification with threshold 0.55 for trading signals.
    Calculates PnL from trading signals and computes Sharpe Ratio.
    """
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, close_prices: pd.Series, n_splits: int = 5):
        """
        Initialize objective function.
        
        Args:
            X: Feature matrix
            y: Target labels (1 for LONG, 0 for NEUTRAL/SHORT)
            close_prices: Close prices for PnL calculation
            n_splits: Number of folds for TimeSeriesSplit
        """
        self.X = X
        self.y = y
        self.close_prices = close_prices
        self.n_splits = n_splits
        
        # Initialize TimeSeriesSplit (no shuffling)
        self.cv = TimeSeriesSplit(n_splits=n_splits)
        
    def calculate_strategy_returns(self, y_pred_proba: pd.Series, close_prices: pd.Series) -> pd.Series:
        """
        Calculate strategy returns based on trading signals.
        
        Simple strategy: Enter LONG position if probability > 0.55
        Hold for 1 period, exit at next close.
        
        Args:
            y_pred_proba: Predicted probabilities for class 1 (LONG)
            close_prices: Close prices
            
        Returns:
            Series of strategy returns
        """
        # Align indices
        common_idx = y_pred_proba.index.intersection(close_prices.index)
        y_pred_proba = y_pred_proba.loc[common_idx]
        close_prices = close_prices.loc[common_idx]
        
        # Generate trading signals: 1 if prob > 0.55, else 0
        signals = (y_pred_proba > 0.55).astype(int)
        
        # Calculate returns: future return if signal is 1, else 0
        future_returns = close_prices.pct_change().shift(-1)  # Next period return
        
        # Strategy returns: signal * future_return
        strategy_returns = signals * future_returns
        
        return strategy_returns.dropna()
    
    def __call__(self, trial: Trial) -> float:
        """
        Objective function for Optuna.
        
        Args:
            trial: Optuna trial
            
        Returns:
            Average Sharpe Ratio across all folds (to be maximized)
        """
        # Suggest hyperparameters with strict constraints (from requirements)
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'gamma': trial.suggest_float('gamma', 0.1, 5.0),
            'reg_alpha': 0.0,  # Fixed for simplicity
            'reg_lambda': 1.0,  # Fixed L2
            'min_child_weight': 1,  # Fixed
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'objective': 'binary:logistic'  # Binary classification
        }
        
        sharpe_ratios = []
        
        # TimeSeriesSplit cross-validation
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(self.X)):
            # Split data
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            val_close_prices = self.close_prices.iloc[val_idx]
            
            # Create and train model
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
            
            # Predict probabilities on validation set
            y_pred_proba = model.predict_proba(X_val)[:, 1]  # Probability of class 1 (LONG)
            y_pred_proba_series = pd.Series(y_pred_proba, index=X_val.index)
            
            # Calculate strategy returns
            strategy_returns = self.calculate_strategy_returns(y_pred_proba_series, val_close_prices)
            
            # Calculate Sharpe Ratio
            if len(strategy_returns) > 1:
                sharpe = calculate_sharpe_ratio(strategy_returns)
                sharpe_ratios.append(sharpe)
            else:
                sharpe_ratios.append(0.0)
            
            # Pruning after first fold
            if fold == 0:
                trial.report(sharpe_ratios[-1], fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        # Return average Sharpe Ratio across folds
        avg_sharpe = np.mean(sharpe_ratios) if sharpe_ratios else 0.0
        return avg_sharpe


class AdvancedHyperparameterOptimizer3Y:
    """
    Advanced hyperparameter optimizer for XGBoost trading models - 3 Year Version.
    
    Focuses on Sharpe Ratio optimization with strict regularization
    to prevent overfitting. Saves to user_data/model_best_params_3y.json
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.study = None
        self.best_params = None
        self.best_value = None
        self.trial_results = []
        
        # Create output directories
        self.output_dir = Path("user_data")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_and_prepare_data(self, 
                             symbol: str = "BTC/USDT",
                             timeframe: str = "5m",
                             days: int = 1095) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Load and prepare data for optimization.
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            days: Number of days to load (1095 for 3 years)
            
        Returns:
            Tuple of (X, y, close_prices) - features, labels, and close prices
        """
        logger.info(f"Loading {days} days of data for {symbol} {timeframe}")
        
        # Calculate start date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Load OHLCV data
        df = get_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            exchange="binance",
            use_cache=True
        )
        
        logger.info(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        
        # Feature engineering
        logger.info("Generating features...")
        feature_engineer = FeatureEngineer()
        X = feature_engineer.fit_transform(df)
        
        # Labeling
        logger.info("Generating labels...")
        labeler = TripleBarrierLabeler(config=TripleBarrierConfig())
        y = labeler.label(df)
        
        # Extract close prices
        close_prices = df['close']
        
        # Align features, labels, and close prices
        common_index = X.index.intersection(y.index).intersection(close_prices.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        close_prices = close_prices.loc[common_index]
        
        # Convert labels to binary: 1 for LONG (original 1), 0 for NEUTRAL/SHORT (-1, 0)
        y_binary = (y == 1).astype(int)
        
        # Remove rows with NaN
        nan_mask = y_binary.isna() | close_prices.isna()
        if nan_mask.any():
            logger.info(f"Removing {nan_mask.sum()} rows with NaN values")
            X = X[~nan_mask]
            y_binary = y_binary[~nan_mask]
            close_prices = close_prices[~nan_mask]
        
        logger.info(f"Final dataset: {len(X)} samples, {X.shape[1]} features")
        logger.info(f"Binary label distribution: {y_binary.value_counts().to_dict()}")
        
        return X, y_binary, close_prices
    
    def optimize(self, 
                 X: pd.DataFrame, 
                 y: pd.Series,
                 close_prices: pd.Series,
                 n_trials: int = 300,
                 timeout: Optional[int] = None,
                 n_jobs: int = -1) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            X: Features
            y: Binary labels
            close_prices: Close prices for PnL calculation
            n_trials: Number of Optuna trials
            timeout: Optimization timeout in seconds
            n_jobs: Number of parallel jobs (-1 for all cores)
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        logger.info(f"Using {n_jobs} parallel jobs")
        
        # Create objective function
        objective = SharpeRatioObjective(X, y, close_prices, n_splits=5)
        
        # Create study with MedianPruner
        self.study = optuna.create_study(
            direction="maximize",
            study_name="xgboost_sharpe_optimization_3y",
            pruner=MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=1,
                interval_steps=1
            )
        )
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
            n_jobs=n_jobs
        )
        
        # Get best results
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        logger.info(f"Optimization completed. Best Sharpe Ratio: {self.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        # Train final model with best parameters
        final_model = self._train_final_model(X, y, self.best_params)
        
        return {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "study": self.study,
            "final_model": final_model,
            "timestamp": datetime.now().isoformat()
        }
    
    def _train_final_model(self, X: pd.DataFrame, y: pd.Series, 
                          params: Dict) -> xgb.XGBClassifier:
        """
        Train final model with best parameters on full dataset.
        
        Args:
            X: Features
            y: Labels
            params: Best hyperparameters
            
        Returns:
            Trained XGBoost model
        """
        logger.info("Training final model with best parameters...")
        
        # Prepare parameters for final training
        final_params = params.copy()
        final_params['n_estimators'] = 800  # Use max from search space
        final_params['n_jobs'] = -1
        
        # Train on full dataset
        model = xgb.XGBClassifier(**final_params)
        model.fit(X, y)
        
        logger.info("Final model trained successfully")
        return model
    
    def save_results(self, results: Dict[str, Any]):
        """
        Save optimization results to disk.
        
        Args:
            results: Optimization results
        """
        # Save best parameters to user_data directory with 3y suffix
        params_path = self.output_dir / "model_best_params_3y.json"
        with open(params_path, 'w') as f:
            json.dump(results["best_params"], f, indent=2)
        logger.info(f"Best parameters saved to {params_path}")
        
        # Save full results to user_data directory
        results_path = self.output_dir / "hyperopt_results_3y.json"
        
        # Convert study to serializable format
        serializable_results = {
            "best_params": results["best_params"],
            "best_value": results["best_value"],
            "timestamp": results["timestamp"],
            "n_trials_completed": len(results["study"].trials),
            "trials_summary": [
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params,
                    "state": str(t.state)
                }
                for t in results["study"].trials
            ]
        }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Full results saved to {results_path}")
        
        # Save importance plot if possible
        try:
            self._plot_importance(results["final_model"])
        except Exception as e:
            logger.warning(f"Could not create importance plot: {e}")
    
    def _plot_importance(self, model: xgb.XGBClassifier):
        """
        Plot and save feature importance.
        
        Args:
            model: Trained XGBoost model
        """
        import matplotlib.pyplot as plt
        
        # Get feature importance
        importance = model.feature_importances_
        feature_names = model.get_booster().feature_names
        
        if feature_names is None:
            logger.warning("No feature names available for importance plot")
            return
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(20)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importance (3-Year Model)')
        plt.tight_layout()
        
        # Save
        plot_path = self.output_dir / "feature_importance_3y.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        logger.info(f"Feature importance plot saved to {plot_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Advanced hyperparameter optimization for XGBoost trading models - 3 Year Version"
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTC/USDT',
        help='Trading pair (default: BTC/USDT)'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        default='5m',
        help='Timeframe (default: 5m)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=1095,
        help='Number of days to load (default: 1095 for 3 years)'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=300,
        help='Number of Optuna trials (default: 300)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Optimization timeout in seconds (default: None)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode (use only recent data for testing)'
    )
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs (-1 for all cores)'
    )
    
    args = parser.parse_args()
    print("\n" + "="*70)
    print("🚀 ADVANCED HYPERPARAMETER OPTIMIZATION - 3 YEAR RETRAINING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Symbol:     {args.symbol}")
    print(f"  Timeframe:  {args.timeframe}")
    print(f"  Days:       {args.days} (3 years)")
    print(f"  Trials:     {args.trials}")
    print(f"  Timeout:    {args.timeout}")
    print(f"  Quick mode: {args.quick}")
    print(f"  N Jobs:     {args.n_jobs}")
    print("\n" + "="*70)
    
    try:
        # Initialize optimizer
        optimizer = AdvancedHyperparameterOptimizer3Y()
        
        # Load and prepare data
        X, y, close_prices = optimizer.load_and_prepare_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            days=args.days
        )
        
        # For quick mode, use only recent data
        if args.quick:
            print("\n⚡ Quick mode: Using only 1000 most recent samples")
            X = X.tail(1000)
            y = y.tail(1000)
            close_prices = close_prices.tail(1000)
            args.trials = min(args.trials, 20)  # Reduce trials for quick mode
        
        # Run optimization
        results = optimizer.optimize(
            X=X,
            y=y,
            close_prices=close_prices,
            n_trials=args.trials,
            timeout=args.timeout,
            n_jobs=args.n_jobs
        )
        
        # Save results
        optimizer.save_results(results)
        
        print("\n" + "="*70)
        print("✅ OPTIMIZATION COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nBest Sharpe Ratio: {results['best_value']:.4f}")
        print(f"\nBest parameters saved to: user_data/model_best_params_3y.json")
        print(f"Full results saved to: user_data/hyperopt_results_3y.json")
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
