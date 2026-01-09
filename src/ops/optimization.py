"""
Freqtrade Optimization Wrapper
==============================

Automates hyperopt and backtesting cycles.
"""

import logging
import subprocess
from pathlib import Path
from typing import Any

from src.config import config

logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """Wrapper for Freqtrade optimization commands."""

    def __init__(
        self,
        data_dir: str | None = None,
        results_dir: str | None = None,
    ):
        cfg = config()
        self.data_dir = Path(data_dir or cfg.paths.data_dir / "binance")
        self.results_dir = Path(results_dir or cfg.paths.user_data_dir / "hyperopt_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_backtest(
        self,
        strategy: str,
        timeframe: str,
        timerange: str | None = None,
        config_path: str | None = None,
    ) -> bool:
        """Run freqtrade backtesting."""
        cmd = [
            "freqtrade",
            "backtesting",
            "--strategy",
            strategy,
            "--timeframe",
            timeframe,
            "--config",
            config_path or str(config().paths.user_data_dir / "config/config_backtest.json"),
        ]

        if timerange:
            cmd.extend(["--timerange", timerange])

        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Backtest failed: {e}")
            return False

    def run_hyperopt(
        self,
        strategy: str,
        timeframe: str,
        epochs: int = 100,
        spaces: list[str] | None = None,
    ) -> bool:
        """Run freqtrade hyperopt."""
        cmd = [
            "freqtrade",
            "hyperopt",
            "--strategy",
            strategy,
            "--timeframe",
            timeframe,
            "--epochs",
            str(epochs),
            "--config",
            str(config().paths.user_data_dir / "config/config_backtest.json"),
        ]

        if spaces:
            cmd.extend(["--spaces"] + spaces)

        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Hyperopt failed: {e}")
            return False
