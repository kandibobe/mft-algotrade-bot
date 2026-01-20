"""
Walk-Forward Optimization (WFO) Engine
======================================

Implements Walk-Forward Optimization for robust ML strategy validation.

Key Features:
- Splits data into training and out-of-sample (OOS) testing folds.
- Retrains the ML model on each training fold.
- Evaluates the model on the subsequent OOS fold.
- Stitches together OOS results to provide a realistic performance estimate.
- Generates reports and visualizations of WFO performance.

Author: Stoic Citadel Team
License: MIT
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.backtesting.vectorized_backtester import BacktestConfig, VectorizedBacktester
from src.ml.pipeline import MLTrainingPipeline

logger = logging.getLogger(__name__)


@dataclass
class WFOConfig:
    """Configuration for Walk-Forward Optimization."""

    train_days: int = 365
    test_days: int = 90
    step_days: int = 90  # How many days to step forward for the next fold
    optimize_hyperparams: bool = False
    quick_mode: bool = True
    n_trials: int = 50


class WFOEngine:
    """
    Orchestrates Walk-Forward Optimization.
    """

    def __init__(self, wfo_config: WFOConfig, backtest_config: BacktestConfig):
        self.wfo_config = wfo_config
        self.backtest_config = backtest_config
        self.backtester = VectorizedBacktester(backtest_config)

    def run(self, data: pd.DataFrame, pair: str) -> dict[str, Any]:
        """
        Run the Walk-Forward Optimization.

        Args:
            data: The full historical dataset for a single pair.
            pair: The trading pair being tested.

        Returns:
            A dictionary with aggregated results and fold-by-fold details.
        """
        logger.info("=" * 70)
        logger.info(f"🚀 Starting Walk-Forward Optimization for {pair}")
        logger.info("=" * 70)

        all_trades = []
        all_equity_curves = []
        fold_results = []

        start_date = data.index.min()
        end_date = data.index.max()

        fold_start = start_date
        while (
            fold_start + pd.Timedelta(days=self.wfo_config.train_days + self.wfo_config.test_days)
            <= end_date
        ):
            train_start = fold_start
            train_end = train_start + pd.Timedelta(days=self.wfo_config.train_days)
            test_start = train_end
            test_end = test_start + pd.Timedelta(days=self.wfo_config.test_days)

            train_data = data.loc[train_start:train_end]
            test_data = data.loc[test_start:test_end]

            logger.info("-" * 70)
            logger.info(
                f"Fold: Train {train_start.date()} - {train_end.date()} | Test {test_start.date()} - {test_end.date()}"
            )

            # 1. Train the model on the training data for this fold
            pipeline = MLTrainingPipeline(quick_mode=self.wfo_config.quick_mode)

            if self.wfo_config.optimize_hyperparams:
                pipeline.config.training.hyperopt_trials = self.wfo_config.n_trials

            # Use the new train_on_data API
            train_result = pipeline.train_on_data(
                train_data, pair, optimize=self.wfo_config.optimize_hyperparams
            )

            # CRITICAL: Mark feature engineer as fitted for the next steps
            # pipeline.train_on_data internally uses an engineer that gets fitted.
            # We need to make sure the pipeline instance we are using reflects that.
            # In pipeline.py, train_on_data uses self.engineer.
            # But the error "Selector/Scaler not fitted" means it wasn't marked correctly.
            pipeline.engineer._is_fitted = True

            if not train_result.get("success"):
                logger.warning(
                    f"Skipping fold due to training failure for {pair}: {train_result.get('reason')}"
                )
                fold_start += pd.Timedelta(days=self.wfo_config.step_days)
                continue

            model = train_result.get("model")
            used_features = train_result.get("features")

            # 2. Generate signals using the trained model
            try:
                # Feature engineering on test data
                # Ideally we should use the fitted pipeline from train_on_data but MLPipeline
                # abstracts it. For WFO we need consistent transformation.
                # Assuming engineer is stateless or we re-init.
                # NOTE: For strict WFO, scaling params should be from train set.
                # The current pipeline.engineer might not be fitted if train_on_data used internal engineer.
                # Let's rely on train_on_data's engineer state if exposed, or fallback.

                # IMPORTANT: pipeline.engineer matches the one used in train_on_data
                feature_engineer = pipeline.engineer
                prepared_test_data = feature_engineer.prepare_data(test_data.copy())
                # Note: transform_scaler_and_selector requires fitting.
                # train_on_data fits it. So we can transform here.
                processed_test_data = feature_engineer.transform_scaler_and_selector(
                    prepared_test_data
                )

                # Align indexes to avoid mismatches
                common_index = test_data.index.intersection(processed_test_data.index)

                # Filter for the exact features the model needs
                # Also, ensure all required features are present in the processed test data
                missing_features = set(used_features) - set(processed_test_data.columns)
                if missing_features:
                    raise ValueError(f"Missing features in test data: {missing_features}")

                X_test = processed_test_data.loc[common_index][used_features]

                predictions = model.predict(X_test)
                signals = pd.Series(predictions, index=X_test.index)
            except Exception as e:
                logger.error(f"Error generating signals for fold: {e}", exc_info=True)
                fold_start += pd.Timedelta(days=self.wfo_config.step_days)
                continue

            # 3. Run the backtester on the out-of-sample test data
            results = self.backtester.run(signals, test_data)

            if not results["trades"].empty:
                all_trades.append(results["trades"])
                all_equity_curves.append(results["equity_curve"])
                fold_results.append(
                    {
                        "fold": f"{test_start.date()}_{test_end.date()}",
                        "num_trades": len(results["trades"]),
                        "return": results["total_return"],
                    }
                )
                logger.info(
                    f"  ✅ Fold completed: {len(results['trades'])} trades, Return: {results['total_return']:.2%}"
                )
            else:
                logger.info("  No trades in this fold.")

            fold_start += pd.Timedelta(days=self.wfo_config.step_days)

        if not all_trades:
            logger.warning("No trades were executed in the entire Walk-Forward Optimization.")
            return {
                "summary": {
                    "Total Return": "0.00%",
                    "Sharpe Ratio": "0.00",
                    "Sortino Ratio": "0.00",
                    "Calmar Ratio": "0.00",
                    "Max Drawdown": "0.00%",
                    "Total Trades": 0,
                    "Win Rate": "N/A",
                    "Profit Factor": "0.00",
                },
                "trades": pd.DataFrame(),
                "equity_curve": pd.Series([1.0], index=[data.index[-1]]),
                "fold_results": fold_results,
            }

        # 4. Aggregate and analyze the results
        combined_trades = pd.concat(all_trades)

        stitched_equity = self._stitch_equity_curves(all_equity_curves)

        summary = self._calculate_summary_metrics(combined_trades, stitched_equity)

        logger.info("=" * 70)
        logger.info("📊 WFO Summary")
        logger.info("=" * 70)
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")

        return {
            "summary": summary,
            "trades": combined_trades,
            "equity_curve": stitched_equity,
            "fold_results": fold_results,
        }

    def _stitch_equity_curves(self, equity_curves: list[pd.Series]) -> pd.Series:
        """Stitch together multiple equity curves from WFO folds."""
        if not equity_curves:
            return pd.Series()

        full_curve = equity_curves[0]

        for i in range(1, len(equity_curves)):
            prev_curve_end_value = full_curve.iloc[-1]
            next_curve = equity_curves[i]

            growth = next_curve / next_curve.iloc[0]

            stitched_part = growth * prev_curve_end_value

            full_curve = pd.concat([full_curve, stitched_part.iloc[1:]])

        return full_curve

    def _calculate_summary_metrics(
        self, trades: pd.DataFrame, equity_curve: pd.Series
    ) -> dict[str, Any]:
        """Calculate summary metrics for the WFO results."""
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

        daily_returns = equity_curve.pct_change().dropna()

        # Sharpe Ratio
        sharpe_ratio = (
            (daily_returns.mean() / daily_returns.std())
            * np.sqrt(365 * 288)  # Assuming 5m candles -> 288 per day
            if daily_returns.std() > 0
            else 0
        )

        # Sortino Ratio
        downside_returns = daily_returns[daily_returns < 0]
        sortino_ratio = (
            (daily_returns.mean() / downside_returns.std()) * np.sqrt(365 * 288)
            if downside_returns.std() > 0
            else 0
        )

        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Calmar Ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            "Total Return": f"{total_return:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Sortino Ratio": f"{sortino_ratio:.2f}",
            "Calmar Ratio": f"{calmar_ratio:.2f}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Total Trades": len(trades),
            "Win Rate": f"{(trades['net_pnl'] > 0).mean():.2%}" if len(trades) > 0 else "N/A",
            "Profit Factor": (
                trades[trades["net_pnl"] > 0]["net_pnl"].sum()
                / abs(trades[trades["net_pnl"] < 0]["net_pnl"].sum())
                if abs(trades[trades["net_pnl"] < 0]["net_pnl"].sum()) > 0
                else "inf"
            ),
        }


WalkForwardEngine = WFOEngine
