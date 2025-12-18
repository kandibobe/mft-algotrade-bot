#!/usr/bin/env python
"""
Walk-Forward Validation
=======================

The ONLY way to get realistic backtest results for trading strategies.

Problem: Standard backtests optimize on entire dataset = overfitting = bad live results.

Solution: Walk-Forward Validation:
1. Split data into N windows (e.g., 5 windows of 6 months each)
2. For each window:
   - Train on window i
   - Test on window i+1
3. Aggregate out-of-sample results

This simulates what would actually happen if you trained monthly and traded the next month.

Usage:
    python scripts/walk_forward_validation.py --pairs BTC/USDT --timeframe 1h --windows 5

    # Or via Makefile:
    make walk-forward
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import json

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.training import (
    FeatureEngineer,
    FeatureConfig,
    ModelTrainer,
    TrainingConfig,
    TripleBarrierLabeler,
    TripleBarrierConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""

    # Data settings
    data_path: str = "user_data/data"
    pairs: List[str] = field(default_factory=lambda: ["BTC/USDT"])
    timeframe: str = "1h"

    # Window settings
    n_windows: int = 5  # Number of train/test windows
    train_ratio: float = 0.8  # 80% train, 20% test within each window

    # ML settings
    model_type: str = "lightgbm"
    optimize_hyperparams: bool = False  # Disable for faster runs

    # Labeling
    take_profit_pct: float = 0.02
    stop_loss_pct: float = 0.01
    max_holding_period: int = 24

    # Output
    output_path: str = "user_data/walk_forward_results"


@dataclass
class WindowResult:
    """Results for a single walk-forward window."""

    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    # Metrics
    train_accuracy: float = 0.0
    test_accuracy: float = 0.0
    train_f1: float = 0.0
    test_f1: float = 0.0
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0

    # Trade statistics
    n_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0


class WalkForwardValidator:
    """
    Walk-Forward Validation for trading strategies.

    This is the gold standard for backtesting ML trading strategies.
    It prevents overfitting by strictly separating train and test periods.

    How it works:
    1. Divide data into N windows
    2. For each window i (except last):
       - Train model on window i
       - Test on window i+1
    3. Aggregate out-of-sample results

    This simulates real trading where you:
    - Train on historical data
    - Trade on future (unseen) data
    - Retrain periodically
    """

    def __init__(self, config: Optional[WalkForwardConfig] = None):
        """Initialize validator."""
        self.config = config or WalkForwardConfig()
        self.results: List[WindowResult] = []

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run walk-forward validation.

        Args:
            df: OHLCV DataFrame with datetime index

        Returns:
            Dictionary with aggregated results
        """
        logger.info(f"Starting Walk-Forward Validation with {self.config.n_windows} windows")
        logger.info(f"Data range: {df.index.min()} to {df.index.max()}")

        # Create windows
        windows = self._create_windows(df)
        logger.info(f"Created {len(windows)} train/test windows")

        # Run each window
        self.results = []

        for window_id, (train_df, test_df) in enumerate(windows):
            logger.info(f"\n{'='*60}")
            logger.info(f"Window {window_id + 1}/{len(windows)}")
            logger.info(f"Train: {train_df.index.min()} to {train_df.index.max()} ({len(train_df)} samples)")
            logger.info(f"Test:  {test_df.index.min()} to {test_df.index.max()} ({len(test_df)} samples)")

            result = self._run_window(window_id, train_df, test_df)
            self.results.append(result)

            logger.info(f"Train Accuracy: {result.train_accuracy:.2%}")
            logger.info(f"Test Accuracy:  {result.test_accuracy:.2%}")
            logger.info(f"Test Sharpe:    {result.test_sharpe:.2f}")

        # Aggregate results
        aggregated = self._aggregate_results()

        logger.info(f"\n{'='*60}")
        logger.info("WALK-FORWARD VALIDATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Average Test Accuracy: {aggregated['avg_test_accuracy']:.2%}")
        logger.info(f"Average Test Sharpe:   {aggregated['avg_test_sharpe']:.2f}")
        logger.info(f"Std Test Accuracy:     {aggregated['std_test_accuracy']:.2%}")

        # Save results
        self._save_results(aggregated)

        return aggregated

    def _create_windows(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create train/test windows for walk-forward validation.

        Returns:
            List of (train_df, test_df) tuples
        """
        n = len(df)
        window_size = n // self.config.n_windows

        windows = []

        for i in range(self.config.n_windows - 1):
            # Train window
            train_start = i * window_size
            train_end = (i + 1) * window_size

            # Test window (next segment)
            test_start = train_end
            test_end = min((i + 2) * window_size, n)

            train_df = df.iloc[train_start:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()

            windows.append((train_df, test_df))

        return windows

    def _run_window(
        self,
        window_id: int,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> WindowResult:
        """
        Run training and testing for a single window.

        Args:
            window_id: Window identifier
            train_df: Training data
            test_df: Testing data

        Returns:
            WindowResult with metrics
        """
        result = WindowResult(
            window_id=window_id,
            train_start=train_df.index.min(),
            train_end=train_df.index.max(),
            test_start=test_df.index.min(),
            test_end=test_df.index.max(),
        )

        try:
            # 1. Create labels with Triple Barrier
            labeler = TripleBarrierLabeler(TripleBarrierConfig(
                take_profit_pct=self.config.take_profit_pct,
                stop_loss_pct=self.config.stop_loss_pct,
                max_holding_period=self.config.max_holding_period,
            ))

            train_labeled = labeler.create_labels(train_df)
            test_labeled = labeler.create_labels(test_df)

            # 2. Feature engineering (fit on train, transform test)
            engineer = FeatureEngineer(FeatureConfig(
                scale_features=True,
                scaling_method="standard",
            ))

            train_features = engineer.fit_transform(train_labeled)
            test_features = engineer.transform(test_labeled)

            # 3. Prepare X, y
            feature_cols = engineer.get_feature_names()

            X_train = train_features[feature_cols].dropna()
            y_train = train_features.loc[X_train.index, "label"]

            X_test = test_features[feature_cols].dropna()
            y_test = test_features.loc[X_test.index, "label"]

            if len(X_train) < 100 or len(X_test) < 20:
                logger.warning(f"Window {window_id}: Insufficient data after cleaning")
                return result

            # 4. Train model
            trainer = ModelTrainer(TrainingConfig(
                model_type=self.config.model_type,
                optimize_hyperparams=self.config.optimize_hyperparams,
            ))

            model, train_metrics = trainer.train(X_train, y_train)

            # 5. Evaluate on test
            test_predictions = model.predict(X_test)
            test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            # Calculate metrics
            from sklearn.metrics import accuracy_score, f1_score

            result.train_accuracy = train_metrics.get("accuracy", 0)
            result.train_f1 = train_metrics.get("f1", 0)
            result.test_accuracy = accuracy_score(y_test, test_predictions)
            result.test_f1 = f1_score(y_test, test_predictions, average="weighted")

            # 6. Calculate trading metrics
            trade_metrics = self._calculate_trading_metrics(
                test_df, y_test, test_predictions
            )
            result.n_trades = trade_metrics["n_trades"]
            result.win_rate = trade_metrics["win_rate"]
            result.profit_factor = trade_metrics["profit_factor"]
            result.total_return = trade_metrics["total_return"]
            result.max_drawdown = trade_metrics["max_drawdown"]
            result.test_sharpe = trade_metrics["sharpe"]

        except Exception as e:
            logger.error(f"Window {window_id} failed: {e}")

        return result

    def _calculate_trading_metrics(
        self,
        df: pd.DataFrame,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate trading performance metrics.

        Args:
            df: OHLCV data
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary with trading metrics
        """
        # Simulate trades based on predictions
        returns = df.loc[y_true.index, "close"].pct_change()

        # Strategy returns: long when pred=1, flat otherwise
        strategy_returns = returns * (y_pred == 1)

        # Remove NaN
        strategy_returns = strategy_returns.dropna()

        if len(strategy_returns) == 0:
            return {
                "n_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe": 0.0,
            }

        # Trades (entries)
        n_trades = (y_pred == 1).sum()

        # Win rate
        winning_trades = strategy_returns[strategy_returns > 0]
        losing_trades = strategy_returns[strategy_returns < 0]
        win_rate = len(winning_trades) / max(1, len(winning_trades) + len(losing_trades))

        # Profit factor
        gross_profit = winning_trades.sum()
        gross_loss = abs(losing_trades.sum())
        profit_factor = gross_profit / max(0.0001, gross_loss)

        # Total return
        total_return = (1 + strategy_returns).prod() - 1

        # Max drawdown
        cumulative = (1 + strategy_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())

        # Sharpe ratio (annualized, assuming hourly data)
        if strategy_returns.std() > 0:
            sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252 * 24)
        else:
            sharpe = 0.0

        return {
            "n_trades": n_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
        }

    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results across all windows."""
        if not self.results:
            return {}

        test_accuracies = [r.test_accuracy for r in self.results]
        test_sharpes = [r.test_sharpe for r in self.results]
        test_returns = [r.total_return for r in self.results]

        return {
            # Accuracy
            "avg_test_accuracy": np.mean(test_accuracies),
            "std_test_accuracy": np.std(test_accuracies),
            "min_test_accuracy": np.min(test_accuracies),
            "max_test_accuracy": np.max(test_accuracies),

            # Sharpe
            "avg_test_sharpe": np.mean(test_sharpes),
            "std_test_sharpe": np.std(test_sharpes),

            # Returns
            "avg_return": np.mean(test_returns),
            "total_return": np.prod([1 + r for r in test_returns]) - 1,

            # Other
            "n_windows": len(self.results),
            "window_results": [
                {
                    "window_id": r.window_id,
                    "train_period": f"{r.train_start} to {r.train_end}",
                    "test_period": f"{r.test_start} to {r.test_end}",
                    "test_accuracy": r.test_accuracy,
                    "test_sharpe": r.test_sharpe,
                    "total_return": r.total_return,
                    "max_drawdown": r.max_drawdown,
                }
                for r in self.results
            ],
        }

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save results to file."""
        output_dir = Path(self.config.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"walk_forward_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_file}")

    def generate_report(self) -> str:
        """Generate human-readable report."""
        if not self.results:
            return "No results to report"

        report = []
        report.append("=" * 70)
        report.append("WALK-FORWARD VALIDATION REPORT")
        report.append("=" * 70)
        report.append("")
        report.append(f"Configuration:")
        report.append(f"  Windows:        {self.config.n_windows}")
        report.append(f"  Model:          {self.config.model_type}")
        report.append(f"  Take Profit:    {self.config.take_profit_pct:.1%}")
        report.append(f"  Stop Loss:      {self.config.stop_loss_pct:.1%}")
        report.append("")
        report.append("Window Results:")
        report.append("-" * 70)

        for r in self.results:
            report.append(
                f"  Window {r.window_id + 1}: "
                f"Acc={r.test_accuracy:.1%} | "
                f"Sharpe={r.test_sharpe:.2f} | "
                f"Return={r.total_return:.1%} | "
                f"DD={r.max_drawdown:.1%}"
            )

        report.append("-" * 70)
        report.append("")

        agg = self._aggregate_results()
        report.append("Aggregated Results:")
        report.append(f"  Average Test Accuracy:  {agg['avg_test_accuracy']:.2%} (+/- {agg['std_test_accuracy']:.2%})")
        report.append(f"  Average Test Sharpe:    {agg['avg_test_sharpe']:.2f} (+/- {agg['std_test_sharpe']:.2f})")
        report.append(f"  Combined Return:        {agg['total_return']:.2%}")
        report.append("")

        # Interpretation
        report.append("Interpretation:")
        if agg['avg_test_accuracy'] > 0.55 and agg['avg_test_sharpe'] > 0.5:
            report.append("  ✅ Strategy shows positive out-of-sample performance")
            report.append("  ✅ Results are likely reproducible in live trading")
        elif agg['avg_test_accuracy'] > 0.50:
            report.append("  ⚠️  Marginal edge detected - proceed with caution")
            report.append("  ⚠️  Consider more data or different features")
        else:
            report.append("  ❌ No significant edge detected")
            report.append("  ❌ Strategy may be overfitted or market is efficient")

        report.append("")
        report.append("=" * 70)

        return "\n".join(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Walk-Forward Validation for Trading Strategies"
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=["BTC/USDT"],
        help="Trading pairs"
    )
    parser.add_argument(
        "--timeframe",
        default="1h",
        help="Timeframe"
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=5,
        help="Number of walk-forward windows"
    )
    parser.add_argument(
        "--model",
        default="lightgbm",
        choices=["lightgbm", "xgboost", "random_forest"],
        help="Model type"
    )
    parser.add_argument(
        "--data-path",
        default="user_data/data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run hyperparameter optimization (slower)"
    )

    args = parser.parse_args()

    # Load data
    from src.data.loader import get_ohlcv

    pair = args.pairs[0].replace("/", "_")
    data_path = Path(args.data_path) / "binance"

    # Try to find data file
    possible_paths = [
        data_path / f"{pair}-{args.timeframe}.feather",
        data_path / f"{pair}-{args.timeframe}.json",
        data_path / f"{pair}-{args.timeframe}.csv",
    ]

    df = None
    for path in possible_paths:
        if path.exists():
            logger.info(f"Loading data from {path}")
            if path.suffix == ".feather":
                df = pd.read_feather(path)
            elif path.suffix == ".json":
                df = pd.read_json(path)
            else:
                df = pd.read_csv(path)
            break

    if df is None:
        logger.error(f"No data found for {args.pairs[0]} {args.timeframe}")
        logger.error(f"Tried: {possible_paths}")
        sys.exit(1)

    # Ensure datetime index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Run validation
    config = WalkForwardConfig(
        pairs=args.pairs,
        timeframe=args.timeframe,
        n_windows=args.windows,
        model_type=args.model,
        optimize_hyperparams=args.optimize,
        data_path=args.data_path,
    )

    validator = WalkForwardValidator(config)
    results = validator.run(df)

    # Print report
    print(validator.generate_report())


if __name__ == "__main__":
    main()
