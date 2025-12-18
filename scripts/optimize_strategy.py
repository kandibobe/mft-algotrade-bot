#!/usr/bin/env python3
"""
Strategy Optimization Script
==============================

Joint optimization of ML model hyperparameters AND trading strategy parameters
using Optuna to maximize Sharpe Ratio or Sortino Ratio.

Key insight: Optimizing ML model and strategy separately is suboptimal.
The best model parameters depend on trading parameters and vice versa.

This script:
1. Defines search space for both ML and trading parameters
2. Uses cross-validation to prevent overfitting
3. Maximizes risk-adjusted returns (Sharpe/Sortino)
4. Outputs best parameters to JSON

Usage:
    python scripts/optimize_strategy.py --n-trials 100 --timerange 20240101-20240601

Author: Stoic Citadel Team
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("Please install optuna: pip install optuna")
    sys.exit(1)

from src.ml.training.feature_engineering import FeatureEngineer, FeatureConfig
from src.ml.training.labeling import TripleBarrierLabeler, TripleBarrierConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """
    Joint optimization of ML and trading parameters.

    Optimizes:
    - ML parameters: model type, hyperparameters
    - Labeling parameters: take profit, stop loss, holding period
    - Trading parameters: entry threshold, exit parameters

    Objective: Maximize Sharpe Ratio using walk-forward validation.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_metric: str = "sharpe",
        cv_folds: int = 5,
        random_state: int = 42,
    ):
        """
        Initialize optimizer.

        Args:
            data: OHLCV DataFrame
            target_metric: 'sharpe' or 'sortino'
            cv_folds: Number of CV folds
            random_state: Random seed
        """
        self.data = data
        self.target_metric = target_metric
        self.cv_folds = cv_folds
        self.random_state = random_state

        # Results storage
        self.best_params: Dict[str, Any] = {}
        self.best_score: float = float('-inf')
        self.trial_history: list = []

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function.

        Args:
            trial: Optuna trial object

        Returns:
            Metric to maximize (Sharpe or Sortino ratio)
        """
        try:
            # 1. Sample ML hyperparameters
            ml_params = self._sample_ml_params(trial)

            # 2. Sample labeling parameters (Triple Barrier)
            labeling_params = self._sample_labeling_params(trial)

            # 3. Sample trading parameters
            trading_params = self._sample_trading_params(trial)

            # 4. Prepare features
            feature_config = FeatureConfig(
                short_period=ml_params.get('short_period', 14),
                medium_period=ml_params.get('medium_period', 50),
                include_time_features=True,
            )
            engineer = FeatureEngineer(feature_config)
            features_df = engineer.transform(self.data.copy())

            # 5. Create labels using Triple Barrier
            labeling_config = TripleBarrierConfig(
                take_profit=labeling_params['take_profit'],
                stop_loss=labeling_params['stop_loss'],
                max_holding_period=labeling_params['max_holding_period'],
                fee_adjustment=labeling_params['fee_adjustment'],
            )
            labeler = TripleBarrierLabeler(labeling_config)
            labels = labeler.label(features_df)

            # 6. Prepare training data
            feature_names = engineer.get_feature_names()
            valid_idx = ~labels.isna()

            X = features_df.loc[valid_idx, feature_names].dropna()
            y = labels[valid_idx].loc[X.index]

            if len(X) < 100:
                logger.warning("Insufficient data after filtering")
                return float('-inf')

            # 7. Cross-validate with walk-forward
            score = self._walk_forward_validation(
                X, y, ml_params, trading_params
            )

            # Store trial result
            self.trial_history.append({
                'trial': trial.number,
                'score': score,
                'ml_params': ml_params,
                'labeling_params': labeling_params,
                'trading_params': trading_params,
            })

            return score

        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return float('-inf')

    def _sample_ml_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample ML model hyperparameters."""
        model_type = trial.suggest_categorical(
            'model_type', ['random_forest', 'xgboost', 'lightgbm']
        )

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
                'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
            })

        elif model_type == 'xgboost':
            params.update({
                'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 300),
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
                'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('xgb_colsample', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-8, 10.0, log=True),
            })

        elif model_type == 'lightgbm':
            params.update({
                'n_estimators': trial.suggest_int('lgb_n_estimators', 50, 300),
                'max_depth': trial.suggest_int('lgb_max_depth', 3, 10),
                'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('lgb_num_leaves', 20, 150),
                'min_child_samples': trial.suggest_int('lgb_min_child', 5, 100),
                'reg_alpha': trial.suggest_float('lgb_reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('lgb_reg_lambda', 1e-8, 10.0, log=True),
            })

        return params

    def _sample_labeling_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample Triple Barrier labeling parameters."""
        return {
            'take_profit': trial.suggest_float('take_profit', 0.003, 0.02),
            'stop_loss': trial.suggest_float('stop_loss', 0.001, 0.01),
            'max_holding_period': trial.suggest_int('max_holding_period', 6, 48),
            'fee_adjustment': trial.suggest_float('fee_adjustment', 0.0005, 0.002),
        }

    def _sample_trading_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample trading strategy parameters."""
        return {
            'entry_threshold': trial.suggest_float('entry_threshold', 0.5, 0.8),
            'exit_profit': trial.suggest_float('exit_profit', 0.01, 0.05),
            'exit_loss_tolerance': trial.suggest_float('exit_loss', 0.01, 0.05),
            'adx_threshold': trial.suggest_int('adx_threshold', 15, 35),
        }

    def _walk_forward_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ml_params: Dict[str, Any],
        trading_params: Dict[str, Any],
    ) -> float:
        """
        Perform walk-forward validation.

        This is more realistic than standard CV for time series.
        """
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        returns_list = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Train model
            model = self._create_model(ml_params)
            model.fit(X_train, y_train)

            # Get predictions
            y_pred_proba = model.predict_proba(X_test)

            # Simulate trading
            fold_returns = self._simulate_trading(
                y_pred_proba, y_test, trading_params
            )
            returns_list.extend(fold_returns)

        if len(returns_list) < 10:
            return float('-inf')

        returns = np.array(returns_list)

        # Calculate metric
        if self.target_metric == 'sharpe':
            return self._calculate_sharpe(returns)
        else:  # sortino
            return self._calculate_sortino(returns)

    def _create_model(self, params: Dict[str, Any]):
        """Create ML model with given parameters."""
        model_type = params.get('model_type', 'random_forest')

        if model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 10),
                min_samples_split=params.get('min_samples_split', 2),
                min_samples_leaf=params.get('min_samples_leaf', 1),
                random_state=self.random_state,
                n_jobs=-1,
            )

        elif model_type == 'xgboost':
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.1),
                subsample=params.get('subsample', 0.8),
                colsample_bytree=params.get('colsample_bytree', 0.8),
                reg_alpha=params.get('reg_alpha', 0.1),
                reg_lambda=params.get('reg_lambda', 0.1),
                random_state=self.random_state,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss',
            )

        elif model_type == 'lightgbm':
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.1),
                num_leaves=params.get('num_leaves', 31),
                min_child_samples=params.get('min_child_samples', 20),
                reg_alpha=params.get('reg_alpha', 0.1),
                reg_lambda=params.get('reg_lambda', 0.1),
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _simulate_trading(
        self,
        y_pred_proba: np.ndarray,
        y_true: pd.Series,
        trading_params: Dict[str, Any],
    ) -> list:
        """
        Simulate trading based on predictions.

        Returns list of trade returns.
        """
        threshold = trading_params.get('entry_threshold', 0.6)
        exit_profit = trading_params.get('exit_profit', 0.02)
        exit_loss = trading_params.get('exit_loss_tolerance', 0.02)

        returns = []

        # Simple simulation: if prediction > threshold, "trade"
        for i, (proba, true_label) in enumerate(zip(y_pred_proba, y_true)):
            # Get probability of positive class
            if len(proba) > 1:
                buy_prob = proba[1] if 1 in [0, 1] else proba[-1]
            else:
                buy_prob = proba[0]

            if buy_prob > threshold:
                # Simulated trade return based on true outcome
                if true_label == 1:  # Correct prediction
                    trade_return = exit_profit * (buy_prob - 0.5) * 2
                elif true_label == -1:  # Wrong direction
                    trade_return = -exit_loss
                else:  # Hold (no movement)
                    trade_return = -0.001  # Small fee loss

                returns.append(trade_return)

        return returns

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate Sharpe Ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return float('-inf')

        # Annualize (assume hourly data)
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Risk-free rate (assume 0 for crypto)
        sharpe = mean_return / std_return

        # Annualize (sqrt of periods per year)
        # For 5m data: 288 periods/day * 365 days
        annualization = np.sqrt(288 * 365)

        return sharpe * annualization

    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """Calculate Sortino Ratio (only penalizes downside volatility)."""
        if len(returns) == 0:
            return float('-inf')

        mean_return = np.mean(returns)
        negative_returns = returns[returns < 0]

        if len(negative_returns) == 0:
            return float('inf')  # No losses

        downside_std = np.std(negative_returns)
        if downside_std == 0:
            return float('inf')

        sortino = mean_return / downside_std

        # Annualize
        annualization = np.sqrt(288 * 365)

        return sortino * annualization

    def optimize(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Run optimization.

        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds (optional)
            show_progress: Show progress bar

        Returns:
            Best parameters dictionary
        """
        logger.info(f"Starting optimization with {n_trials} trials")
        logger.info(f"Target metric: {self.target_metric.upper()} Ratio")

        sampler = TPESampler(seed=self.random_state)

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name="stoic_citadel_optimization",
        )

        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress,
            catch=(Exception,),
        )

        self.best_params = study.best_params
        self.best_score = study.best_value

        logger.info(f"\nOptimization complete!")
        logger.info(f"Best {self.target_metric} ratio: {self.best_score:.4f}")
        logger.info(f"Best parameters: {json.dumps(self.best_params, indent=2)}")

        return self.best_params

    def save_results(self, output_path: str = "user_data/models/optimized_params.json"):
        """Save optimization results to JSON."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "timestamp": datetime.now().isoformat(),
            "target_metric": self.target_metric,
            "best_score": self.best_score,
            "best_params": self.best_params,
            "cv_folds": self.cv_folds,
            "n_trials": len(self.trial_history),
            "trial_history": self.trial_history[-10:],  # Last 10 trials
        }

        with open(output, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output}")


def load_data(data_path: str) -> pd.DataFrame:
    """Load OHLCV data from various sources."""
    path = Path(data_path)

    if path.suffix == '.csv':
        df = pd.read_csv(path, parse_dates=['date'], index_col='date')
    elif path.suffix == '.feather':
        df = pd.read_feather(path)
        if 'date' in df.columns:
            df.set_index('date', inplace=True)
    elif path.suffix == '.parquet':
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Ensure required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Optimize strategy parameters")
    parser.add_argument(
        "--data", "-d",
        type=str,
        default="user_data/data/binance/BTC_USDT-5m.feather",
        help="Path to OHLCV data file"
    )
    parser.add_argument(
        "--n-trials", "-n",
        type=int,
        default=100,
        help="Number of optimization trials"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds"
    )
    parser.add_argument(
        "--metric", "-m",
        type=str,
        default="sharpe",
        choices=["sharpe", "sortino"],
        help="Optimization metric"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="user_data/models/optimized_params.json",
        help="Output path for results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Load data
    logger.info(f"Loading data from {args.data}")
    try:
        data = load_data(args.data)
        logger.info(f"Loaded {len(data)} rows")
    except FileNotFoundError:
        # Generate sample data for testing
        logger.warning("Data file not found, generating sample data")
        np.random.seed(args.seed)
        n = 5000

        dates = pd.date_range(start='2024-01-01', periods=n, freq='5min')
        close = 50000 + np.cumsum(np.random.randn(n) * 50)

        data = pd.DataFrame({
            'open': close - np.random.uniform(0, 50, n),
            'high': close + np.random.uniform(0, 100, n),
            'low': close - np.random.uniform(0, 100, n),
            'close': close,
            'volume': np.random.randint(100, 10000, n),
        }, index=dates)

    # Run optimization
    optimizer = StrategyOptimizer(
        data=data,
        target_metric=args.metric,
        cv_folds=args.cv_folds,
        random_state=args.seed,
    )

    best_params = optimizer.optimize(
        n_trials=args.n_trials,
        timeout=args.timeout,
    )

    # Save results
    optimizer.save_results(args.output)

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best {args.metric.upper()} Ratio: {optimizer.best_score:.4f}")
    print(f"Results saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
