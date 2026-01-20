"""
Meta-Labeling Utility
======================

Implements the Meta-Labeling technique (De Prado).
Meta-labeling is a secondary ML model that predicts the probability of success
of a primary model's signal.

Primary Model: Is this a BUY or SELL? (Signal generation)
Secondary Model (Meta): Is the Primary Model likely to be right? (Binary filtering)
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MetaModel:
    """
    Wraps a secondary ML model for meta-labeling.
    """

    def __init__(self, model_obj: Any = None):
        self.model = model_obj

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of trade success."""
        if self.model is None:
            return np.full(len(X), 0.5)

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        elif hasattr(self.model, "predict"):
            return self.model.predict(X)
        else:
            return np.full(len(X), 0.5)


class MetaLabeler:
    """
    Utility for creating meta-labels for secondary model training.
    """

    @staticmethod
    def create_meta_labels(trades_df: pd.DataFrame) -> pd.Series:
        """
        Create meta-labels based on trade outcomes.

        A meta-label is 1 if the primary signal was profitable (hit TP),
        and 0 otherwise (hit SL or timed out).

        Args:
            trades_df: DataFrame containing trade execution results
                       Expected columns: 'return_pct', 'barrier_hit'

        Returns:
            Series of binary meta-labels (0 or 1)
        """
        # 1 if trade was profitable (hit take_profit), 0 otherwise
        meta_labels = (trades_df["return_pct"] > 0).astype(int)

        # If we have barrier information, we can be more precise
        if "barrier_hit" in trades_df.columns:
            meta_labels = (trades_df["barrier_hit"] == "take_profit").astype(int)

        logger.info(
            f"Created {len(meta_labels)} meta-labels. Positive rate: {meta_labels.mean():.2%}"
        )

        return meta_labels

    @staticmethod
    def filter_signals(
        signals: pd.Series, meta_predictions: pd.Series, threshold: float = 0.5
    ) -> pd.Series:
        """
        Filter primary signals using meta-model predictions.

        Args:
            signals: Original signals (1 for BUY, 0 for HOLD)
            meta_predictions: Probability of success from meta-model
            threshold: Confidence threshold to keep a signal

        Returns:
            Filtered signals
        """
        filtered = signals.copy()
        # Set signal to 0 if meta-model confidence is below threshold
        filtered[meta_predictions < threshold] = 0

        dropped_count = (signals != filtered).sum()
        logger.info(
            f"Meta-labeling filtered out {dropped_count} signals "
            f"({dropped_count / max(1, len(signals)):.2%})"
        )

        return filtered
