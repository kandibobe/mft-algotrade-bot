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
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.ml.pipeline import MLTrainingPipeline
from src.backtesting.vectorized_backtester import VectorizedBacktester, BacktestConfig

logger = logging.getLogger(__name__)


@dataclass
class WFOConfig:
    """Configuration for Walk-Forward Optimization."""
    train_days: int = 365
    test_days: int = 90
    step_days: int = 90  # How many days to step forward for the next fold


class WFOEngine:
    """
    Orchestrates Walk-Forward Optimization.
    """

    def __init__(self, wfo_config: WFOConfig, backtest_config: BacktestConfig):
        self.wfo_config = wfo_config
        self.backtest_config = backtest_config
        self.backtester = VectorizedBacktester(backtest_config)

    def run(self, data: pd.DataFrame, pair: str) -> Dict[str, Any]:
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
        while fold_start + pd.Timedelta(days=self.wfo_config.train_days + self.wfo_config.test_days) <= end_date:
            train_start = fold_start
            train_end = train_start + pd.Timedelta(days=self.wfo_config.train_days)
            test_start = train_end
            test_end = test_start + pd.Timedelta(days=self.wfo_config.test_days)

            train_data = data.loc[train_start:train_end]
            test_data = data.loc[test_start:test_end]

            logger.info("-" * 70)
            logger.info(f"Fold: Train {train_start.date()} - {train_end.date()} | Test {test_start.date()} - {test_end.date()}")

            # 1. Train the model on the training data for this fold
            pipeline = MLTrainingPipeline(quick_mode=True) # Using quick_mode for speed in this example
            model = pipeline.train_and_get_model(train_data, pair)
            
            if model is None:
                logger.warning(f"Skipping fold due to training failure for {pair}")
                continue

            # 2. Generate signals using the trained model
            try:
                predictions = model.predict(test_data)
                signals = pd.Series(predictions, index=test_data.index)
            except Exception as e:
                logger.error(f"Error generating signals for fold: {e}")
                continue

            # 3. Run the backtester on the out-of-sample test data
            results = self.backtester.run(signals, test_data)
            
            if not results['trades'].empty:
                all_trades.append(results['trades'])
                all_equity_curves.append(results['equity_curve'])
                fold_results.append({
                    "fold": f"{test_start.date()}_{test_end.date()}",
                    "num_trades": len(results['trades']),
                    "return": results['total_return']
                })
                logger.info(f"  ✅ Fold completed: {len(results['trades'])} trades, Return: {results['total_return']:.2%}")
            else:
                logger.info("  No trades in this fold.")

            fold_start += pd.Timedelta(days=self.wfo_config.step_days)
            
        if not all_trades:
            logger.warning("No trades were executed in the entire Walk-Forward Optimization.")
            return {"summary": "No trades executed."}

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

    def _stitch_equity_curves(self, equity_curves: List[pd.Series]) -> pd.Series:
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
        
    def _calculate_summary_metrics(self, trades: pd.DataFrame, equity_curve: pd.Series) -> Dict[str, Any]:
        """Calculate summary metrics for the WFO results."""
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        
        daily_returns = equity_curve.pct_change().dropna()
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        return {
            "Total Return": f"{total_return:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Total Trades": len(trades),
            "Win Rate": f"{(trades['net_pnl'] > 0).mean():.2%}" if len(trades) > 0 else "N/A",
            "Profit Factor": (
                trades[trades['net_pnl'] > 0]['net_pnl'].sum() / 
                abs(trades[trades['net_pnl'] < 0]['net_pnl'].sum())
                if abs(trades[trades['net_pnl'] < 0]['net_pnl'].sum()) > 0 else "inf"
            ),
        }
