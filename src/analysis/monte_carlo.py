"""
GPU-Accelerated Monte Carlo Simulator
=====================================

High-performance simulation for strategy robustness testing.
Uses CuPy for vectorized GPU operations and Numba for JIT-optimized kernels.
"""

import logging
import time
from typing import Any

import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from src.utils.logger import log

logger = logging.getLogger(__name__)

class GPUMonteCarloSimulator:
    """
    GPU-accelerated Monte Carlo simulator for strategy backtest results.
    """

    def __init__(
        self,
        profits: np.ndarray,
        iterations: int = 10000,
        initial_capital: float = 10000.0
    ):
        self.profits = profits
        self.iterations = iterations
        self.initial_capital = initial_capital
        self.results = None

    def run(self):
        """Run the simulation."""
        if not GPU_AVAILABLE:
            logger.warning("GPU not available, falling back to CPU vectorized execution.")
            return self._run_cpu()

        start_time = time.time()

        # Move data to GPU
        profits_gpu = cp.array(self.profits)
        n_trades = len(self.profits)

        # Parallel simulation: [iterations, n_trades]
        # Generate indices for shuffling (vectorized)
        indices = cp.random.rand(self.iterations, n_trades).argsort(axis=1)

        # Re-arrange profits based on indices
        shuffled_profits = profits_gpu[indices]

        # Vectorized Pnl calculation
        equity_curves = cp.cumprod(1 + shuffled_profits, axis=1) * self.initial_capital

        # Calculate Max Drawdown for each iteration
        peaks = cp.maximum.accumulate(equity_curves, axis=1)
        drawdowns = (peaks - equity_curves) / peaks
        max_drawdowns = cp.max(drawdowns, axis=1)

        # Calculate Returns
        final_equity = equity_curves[:, -1]
        returns = (final_equity - self.initial_capital) / self.initial_capital

        # Fetch results back to CPU
        self.max_drawdowns = cp.asnumpy(max_drawdowns)
        self.returns = cp.asnumpy(returns)

        duration = time.time() - start_time
        log.info("monte_carlo_gpu_complete",
                 iterations=self.iterations,
                 duration_sec=f"{duration:.4f}s")

        return self._format_results()

    def _run_cpu(self):
        """Vectorized CPU fallback using NumPy."""
        start_time = time.time()
        n_trades = len(self.profits)

        # Memory-efficient chunked execution if iterations are high
        chunk_size = 1000
        all_max_dd = []
        all_returns = []

        for i in range(0, self.iterations, chunk_size):
            actual_chunk = min(chunk_size, self.iterations - i)

            # Vectorized shuffle (sort random floats)
            idx = np.random.rand(actual_chunk, n_trades).argsort(axis=1)
            shuffled = self.profits[idx]

            equity = np.cumprod(1 + shuffled, axis=1) * self.initial_capital
            peaks = np.maximum.accumulate(equity, axis=1)
            max_dd = np.max((peaks - equity) / peaks, axis=1)
            final_ret = (equity[:, -1] - self.initial_capital) / self.initial_capital

            all_max_dd.extend(max_dd)
            all_returns.extend(final_ret)

        self.max_drawdowns = np.array(all_max_dd)
        self.returns = np.array(all_returns)

        duration = time.time() - start_time
        log.info("monte_carlo_cpu_complete",
                 iterations=self.iterations,
                 duration_sec=f"{duration:.4f}s")

        return self._format_results()

    def _format_results(self) -> dict[str, Any]:
        return {
            "mean_drawdown": float(np.mean(self.max_drawdowns)),
            "95th_drawdown": float(np.percentile(self.max_drawdowns, 95)),
            "99th_drawdown": float(np.percentile(self.max_drawdowns, 99)),
            "mean_return": float(np.mean(self.returns)),
            "std_return": float(np.std(self.returns)),
            "prob_ruin": float(np.mean(self.max_drawdowns > 0.5)) * 100
        }
