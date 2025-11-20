#!/usr/bin/env python3
"""
Stoic Citadel - Walk-Forward Optimization Script
================================================

Prevents overfitting by validating strategies on out-of-sample data.

Methodology:
-----------
1. Split historical data into Train and Test windows
2. Optimize strategy parameters on Train data
3. Validate on Test data (never seen during optimization)
4. Roll forward and repeat

This ensures the strategy works on unseen data, not just fitted to history.

Usage:
------
    python scripts/walk_forward.py --strategy StoicStrategyV1 \\
                                   --config user_data/config/config_production.json \\
                                   --train-months 3 \\
                                   --test-months 1

Author: Stoic Citadel Team
Version: 1.0.0
"""

import subprocess
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from dataclasses import dataclass


@dataclass
class WalkForwardWindow:
    """Represents a train/test window for walk-forward analysis."""

    train_start: str
    train_end: str
    test_start: str
    test_end: str
    window_id: int


class WalkForwardOptimizer:
    """
    Orchestrates walk-forward optimization using Freqtrade.
    """

    def __init__(
        self,
        strategy: str,
        config_path: str,
        train_months: int = 3,
        test_months: int = 1,
        hyperopt_epochs: int = 100,
        min_sharpe: float = 1.5,
        max_drawdown: float = 0.20,
    ):
        self.strategy = strategy
        self.config_path = config_path
        self.train_months = train_months
        self.test_months = test_months
        self.hyperopt_epochs = hyperopt_epochs
        self.min_sharpe = min_sharpe
        self.max_drawdown = max_drawdown

        self.results: List[Dict] = []

    def generate_windows(
        self, start_date: datetime, end_date: datetime
    ) -> List[WalkForwardWindow]:
        """
        Generate train/test windows for walk-forward analysis.

        Example with 3-month train, 1-month test:
        Window 1: Train [Jan-Mar], Test [Apr]
        Window 2: Train [Feb-Apr], Test [May]
        Window 3: Train [Mar-May], Test [Jun]
        ...
        """
        windows = []
        window_id = 1

        current_date = start_date

        while current_date + timedelta(days=30 * (self.train_months + self.test_months)) <= end_date:
            train_start = current_date
            train_end = train_start + timedelta(days=30 * self.train_months)
            test_start = train_end
            test_end = test_start + timedelta(days=30 * self.test_months)

            windows.append(
                WalkForwardWindow(
                    train_start=train_start.strftime("%Y%m%d"),
                    train_end=train_end.strftime("%Y%m%d"),
                    test_start=test_start.strftime("%Y%m%d"),
                    test_end=test_end.strftime("%Y%m%d"),
                    window_id=window_id,
                )
            )

            # Roll forward by test_months
            current_date = test_start
            window_id += 1

        return windows

    def run_hyperopt(self, window: WalkForwardWindow) -> bool:
        """
        Run hyperopt on the training window.

        Returns True if optimization completed successfully.
        """
        print(f"\n{'=' * 70}")
        print(f"Window {window.window_id}: HYPEROPT (Training)")
        print(f"{'=' * 70}")
        print(f"Train Period: {window.train_start} - {window.train_end}")
        print(f"Epochs: {self.hyperopt_epochs}")

        timerange = f"{window.train_start}-{window.train_end}"

        cmd = [
            "docker-compose",
            "run",
            "--rm",
            "freqtrade",
            "hyperopt",
            "--hyperopt-loss",
            "SharpeHyperOptLoss",
            "--strategy",
            self.strategy,
            "--config",
            f"/freqtrade/{self.config_path}",
            "--timerange",
            timerange,
            "--epochs",
            str(self.hyperopt_epochs),
            "--spaces",
            "buy",
            "sell",
            "--random-state",
            "42",  # Reproducibility
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=3600
            )

            if "Best result" in result.stdout:
                print("✅ HyperOpt completed successfully")
                return True
            else:
                print("⚠️  HyperOpt completed but no improvement found")
                return False

        except subprocess.TimeoutExpired:
            print("❌ HyperOpt timed out (>1 hour)")
            return False
        except subprocess.CalledProcessError as e:
            print(f"❌ HyperOpt failed: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            return False

    def run_backtest(self, window: WalkForwardWindow, mode: str = "test") -> Dict:
        """
        Run backtest on either train or test window.

        Args:
            window: The walk-forward window
            mode: "train" or "test"

        Returns:
            Dictionary with backtest results
        """
        if mode == "train":
            timerange = f"{window.train_start}-{window.train_end}"
            label = "TRAIN"
        else:
            timerange = f"{window.test_start}-{window.test_end}"
            label = "TEST"

        print(f"\n{'=' * 70}")
        print(f"Window {window.window_id}: BACKTEST ({label})")
        print(f"{'=' * 70}")
        print(f"Period: {timerange}")

        cmd = [
            "docker-compose",
            "run",
            "--rm",
            "freqtrade",
            "backtesting",
            "--strategy",
            self.strategy,
            "--config",
            f"/freqtrade/{self.config_path}",
            "--timerange",
            timerange,
            "--export",
            "trades",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=600
            )

            # Parse backtest results from output
            metrics = self._parse_backtest_output(result.stdout)

            if metrics:
                print(f"\n📊 {label} Results:")
                print(f"   Total Return: {metrics.get('total_return', 'N/A')}")
                print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}")
                print(f"   Max Drawdown: {metrics.get('max_drawdown', 'N/A')}")
                print(f"   Win Rate: {metrics.get('win_rate', 'N/A')}")
                print(f"   Total Trades: {metrics.get('total_trades', 'N/A')}")

            return metrics

        except subprocess.TimeoutExpired:
            print(f"❌ Backtest timed out")
            return {}
        except subprocess.CalledProcessError as e:
            print(f"❌ Backtest failed: {e}")
            return {}

    def _parse_backtest_output(self, output: str) -> Dict:
        """
        Extract key metrics from Freqtrade backtest output.

        This is a simplified parser. In production, use JSON output.
        """
        metrics = {}

        try:
            lines = output.split("\n")

            for line in lines:
                if "Total trades" in line:
                    metrics["total_trades"] = line.split()[-1]
                elif "Win/Loss" in line or "Win  %" in line:
                    # Extract win rate
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "%" in part and i > 0:
                            metrics["win_rate"] = part
                            break
                elif "Avg Profit" in line or "Total profit" in line:
                    parts = line.split()
                    if "%" in line:
                        for part in parts:
                            if "%" in part:
                                metrics["total_return"] = part
                                break
                elif "Max Drawdown" in line or "Drawdown" in line:
                    parts = line.split()
                    if "%" in line:
                        for part in parts:
                            if "%" in part:
                                metrics["max_drawdown"] = part
                                break
                elif "Sharpe" in line:
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            # Extract numeric value
                            for part in parts:
                                if part.replace(".", "").replace("-", "").isdigit():
                                    metrics["sharpe_ratio"] = float(part)
                                    break
                        except:
                            pass

        except Exception as e:
            print(f"Warning: Could not parse backtest output: {e}")

        return metrics

    def evaluate_window(self, test_metrics: Dict) -> Tuple[bool, str]:
        """
        Evaluate if test results meet minimum criteria.

        Returns:
            (passed: bool, reason: str)
        """
        reasons = []

        # Check Sharpe Ratio
        sharpe = test_metrics.get("sharpe_ratio")
        if sharpe is not None:
            try:
                sharpe_val = float(sharpe)
                if sharpe_val < self.min_sharpe:
                    reasons.append(
                        f"Sharpe Ratio {sharpe_val:.2f} < {self.min_sharpe}"
                    )
            except:
                reasons.append("Could not parse Sharpe Ratio")

        # Check Max Drawdown
        drawdown = test_metrics.get("max_drawdown")
        if drawdown:
            try:
                dd_val = float(drawdown.replace("%", "")) / 100
                if abs(dd_val) > self.max_drawdown:
                    reasons.append(
                        f"Max Drawdown {abs(dd_val):.2%} > {self.max_drawdown:.2%}"
                    )
            except:
                reasons.append("Could not parse Max Drawdown")

        # Check total trades (need minimum sample size)
        trades = test_metrics.get("total_trades")
        if trades:
            try:
                trades_val = int(trades)
                if trades_val < 5:
                    reasons.append(f"Insufficient trades: {trades_val} < 5")
            except:
                pass

        if reasons:
            return False, "; ".join(reasons)
        else:
            return True, "All criteria met"

    def run_walk_forward(
        self, start_date: str = "20220101", end_date: str = "20241231"
    ) -> pd.DataFrame:
        """
        Execute complete walk-forward optimization.

        Args:
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format

        Returns:
            DataFrame with results for all windows
        """
        print("\n" + "=" * 70)
        print("STOIC CITADEL - WALK-FORWARD OPTIMIZATION")
        print("=" * 70)
        print(f"Strategy: {self.strategy}")
        print(f"Train Window: {self.train_months} months")
        print(f"Test Window: {self.test_months} months")
        print(f"Min Sharpe Ratio: {self.min_sharpe}")
        print(f"Max Drawdown: {self.max_drawdown:.0%}")
        print("=" * 70)

        # Generate windows
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        windows = self.generate_windows(start_dt, end_dt)

        print(f"\nGenerated {len(windows)} walk-forward windows")

        # Process each window
        for window in windows:
            result = {
                "window_id": window.window_id,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
            }

            # Step 1: Optimize on train data
            hyperopt_success = self.run_hyperopt(window)
            result["hyperopt_success"] = hyperopt_success

            if not hyperopt_success:
                result["status"] = "FAILED"
                result["reason"] = "HyperOpt failed"
                self.results.append(result)
                continue

            # Step 2: Backtest on train data (sanity check)
            train_metrics = self.run_backtest(window, mode="train")
            result.update({f"train_{k}": v for k, v in train_metrics.items()})

            # Step 3: Backtest on test data (out-of-sample)
            test_metrics = self.run_backtest(window, mode="test")
            result.update({f"test_{k}": v for k, v in test_metrics.items()})

            # Step 4: Evaluate results
            passed, reason = self.evaluate_window(test_metrics)
            result["status"] = "PASSED" if passed else "FAILED"
            result["reason"] = reason

            self.results.append(result)

            # Print window summary
            print(f"\n{'=' * 70}")
            print(f"Window {window.window_id} Summary")
            print(f"{'=' * 70}")
            print(f"Status: {'✅ PASSED' if passed else '❌ FAILED'}")
            print(f"Reason: {reason}")
            print(f"{'=' * 70}\n")

        # Create results DataFrame
        results_df = pd.DataFrame(self.results)

        # Print final summary
        self._print_summary(results_df)

        return results_df

    def _print_summary(self, results_df: pd.DataFrame):
        """Print final walk-forward summary."""
        print("\n" + "=" * 70)
        print("WALK-FORWARD OPTIMIZATION - FINAL SUMMARY")
        print("=" * 70)

        total_windows = len(results_df)
        passed_windows = len(results_df[results_df["status"] == "PASSED"])
        pass_rate = (passed_windows / total_windows * 100) if total_windows > 0 else 0

        print(f"Total Windows: {total_windows}")
        print(f"Passed: {passed_windows}")
        print(f"Failed: {total_windows - passed_windows}")
        print(f"Pass Rate: {pass_rate:.1f}%")

        print("\n" + "=" * 70)

        if pass_rate >= 70:
            print("✅ STRATEGY APPROVED")
            print("Strategy shows consistent performance across time periods.")
        elif pass_rate >= 50:
            print("⚠️  STRATEGY MARGINAL")
            print("Strategy shows inconsistent performance. Review failed windows.")
        else:
            print("❌ STRATEGY REJECTED")
            print("Strategy fails on out-of-sample data. High overfitting risk.")

        print("=" * 70 + "\n")

        # Save results
        output_file = f"walk_forward_results_{self.strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Walk-Forward Optimization for Freqtrade Strategies"
    )

    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="Strategy name (e.g., StoicStrategyV1)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="user_data/config/config_production.json",
        help="Path to Freqtrade config",
    )

    parser.add_argument(
        "--train-months",
        type=int,
        default=3,
        help="Training window size in months (default: 3)",
    )

    parser.add_argument(
        "--test-months",
        type=int,
        default=1,
        help="Testing window size in months (default: 1)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="HyperOpt epochs per window (default: 100)",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default="20220101",
        help="Start date YYYYMMDD (default: 20220101)",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default="20241231",
        help="End date YYYYMMDD (default: 20241231)",
    )

    parser.add_argument(
        "--min-sharpe",
        type=float,
        default=1.5,
        help="Minimum Sharpe Ratio for pass (default: 1.5)",
    )

    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=0.20,
        help="Maximum drawdown for pass (default: 0.20)",
    )

    args = parser.parse_args()

    # Create optimizer
    optimizer = WalkForwardOptimizer(
        strategy=args.strategy,
        config_path=args.config,
        train_months=args.train_months,
        test_months=args.test_months,
        hyperopt_epochs=args.epochs,
        min_sharpe=args.min_sharpe,
        max_drawdown=args.max_drawdown,
    )

    # Run walk-forward optimization
    results = optimizer.run_walk_forward(
        start_date=args.start_date, end_date=args.end_date
    )

    # Exit with appropriate code
    passed = len(results[results["status"] == "PASSED"])
    total = len(results)
    pass_rate = (passed / total * 100) if total > 0 else 0

    if pass_rate >= 70:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == "__main__":
    main()
