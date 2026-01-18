"""
Walk-Forward Optimization Automation
====================================

Orchestrates multiple backtesting cycles over different time windows.
"""

import logging
from pathlib import Path

from src.config import config

logger = logging.getLogger(__name__)


class WFOAutomation:
    """Automates Walk-Forward Optimization (WFO)."""

    def __init__(
        self,
        results_dir: str | None = None,
    ):
        cfg = config()
        self.results_dir = Path(results_dir or cfg.paths.user_data_dir / "walk_forward_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_wfo_cycle(self, strategy: str, pairs: list[str], windows: int = 3):
        """Run a full WFO cycle."""
        logger.info(f"Starting WFO cycle for {strategy} with {windows} windows")
        # Logic for splitting timerange and calling StrategyOptimizer
        pass
