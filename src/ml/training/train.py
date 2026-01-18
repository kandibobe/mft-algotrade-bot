"""
ML Training Entry Point
=======================

Standardized interface for training ML models.
Integrates configuration, pipeline execution, and model versioning.
"""

import logging

from src.config.manager import ConfigurationManager
from src.ml.pipeline import MLTrainingPipeline

logger = logging.getLogger(__name__)


def train_model(
    pairs: list[str],
    days: int | None = None,
    target: str | None = None,
    timeframe: str = "5m",
    optimize: bool = False,
    quick: bool = False,
    n_trials: int = 50,
):
    """
    Train ML models for specified pairs.

    Args:
        pairs: List of trading pairs (e.g. ['BTC/USDT'])
        days: Number of days of history to use (None = all)
        target: Target variable name (overrides config)
        timeframe: Candle timeframe (default '5m')
        optimize: Whether to run hyperparameter optimization
        quick: Quick mode (reduced data for testing)
        n_trials: Number of optimization trials (if optimize=True)

    Returns:
        Dict containing training results
    """
    # Load config to override if needed
    config = ConfigurationManager.get_config()

    if target:
        logger.info(f"Overriding target variable to: {target}")
        config.training.target_variable = target

    pipeline = MLTrainingPipeline(quick_mode=quick)

    results = pipeline.run(
        pairs=pairs, timeframe=timeframe, optimize=optimize, n_trials=n_trials, days=days
    )

    return results


if __name__ == "__main__":
    # Basic CLI for standalone execution (though manage.py is preferred)
    import argparse

    parser = argparse.ArgumentParser(description="ML Training")
    parser.add_argument("--pairs", nargs="+", default=["BTC/USDT"], help="Pairs")
    parser.add_argument("--days", type=int, help="Days")
    parser.add_argument("--target", help="Target variable")
    parser.add_argument("--optimize", action="store_true", help="Optimize")

    args = parser.parse_args()
    train_model(args.pairs, days=args.days, target=args.target, optimize=args.optimize)
