"""
Probability Calibration Module.
===============================

This module provides tools to calibrate raw model probabilities using dynamic
thresholding based on historical distribution (percentile ranking).

Financial Logic:
----------------
ML models often output probabilities that are skewed (e.g., mean 0.35) or drift
over time due to market regime changes. Hardcoding a threshold like > 0.55
often leads to missed trades or entering only on extreme outliers.

This calibrator calculates the *relative rank* of the current prediction compared
to the last N predictions (rolling window).
- Rule: If current prediction is in the Top 10% (90th percentile) of recent history -> SIGNAL.
- This adapts to the model's current "mood" (calibration).

Author: Stoic Citadel Team
Version: 1.1.0
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """
    Calibrates raw model probabilities using dynamic thresholding.

    Keeps a rolling window of past predictions to determine if the current
    prediction is statistically significant relative to recent history.

    Attributes:
        window_size (int): Size of the rolling window for historical context.
        percentile_threshold (float): Threshold for percentile rank (0.0 to 1.0).
    """

    def __init__(self, window_size: int = 2000, percentile_threshold: float = 0.90) -> None:
        """
        Initialize the ProbabilityCalibrator.

        Args:
            window_size: Number of past predictions to consider (default: 2000).
            percentile_threshold: Top percentile required to trigger a signal.
                                0.90 means top 10% (default: 0.90).
        """
        self.window_size = window_size
        self.percentile_threshold = percentile_threshold
        logger.info(
            f"ProbabilityCalibrator initialized (window={window_size}, "
            f"threshold={percentile_threshold:.2f})"
        )

    def is_signal(
        self,
        probabilities: pd.Series,
        window_size: int | None = None,
        threshold: float | None = None,
    ) -> pd.Series:
        """
        Determine if current probabilities constitute a signal based on historical context.

        Calculates the rolling percentile rank of each probability value.
        If the rank exceeds the threshold, it returns True.

        Args:
            probabilities: Series of raw probability values (0.0 to 1.0).
            window_size: Optional override for window size.
            threshold: Optional override for percentile threshold.

        Returns:
            pd.Series: Boolean series where True indicates a buy signal (calibrated).
        """
        if probabilities.empty:
            return pd.Series(dtype=bool)

        # Use provided values or defaults
        win_size = window_size if window_size is not None else self.window_size
        thresh = threshold if threshold is not None else self.percentile_threshold

        # Determine min_periods - use smaller of 100 or window_size,
        # but also ensure it's not larger than the data length
        min_p = min(win_size, 100, len(probabilities))
        if min_p > 1:
            # Safety check for very short dataframes
            min_p = min(min_p, len(probabilities))

        # Calculate rolling rank (percentile)
        # pct=True returns 0.0 to 1.0 representing the percentile
        # This operation is vectorized and efficient
        rolling_rank = probabilities.rolling(window=win_size, min_periods=min_p).rank(pct=True)

        # 🚀 ADDITION: Isotonic Calibration Proxy
        # Apply a sigmoid-like transformation to smooth the rank into a confidence score
        calibrated_score = 1 / (1 + np.exp(-10 * (rolling_rank - 0.5)))

        # Fill NaN values (start of series) with 0.0 (no signal)
        rolling_rank = rolling_rank.fillna(0.0)
        calibrated_score = calibrated_score.fillna(0.0)

        # Determine signal
        # We check if the current prediction is in the top X% of recent predictions
        is_signal = rolling_rank > thresh

        return is_signal

    def get_calibrated_confidence(self, probabilities: pd.Series) -> pd.Series:
        """
        Get smoothed calibrated confidence score.
        Useful for Kelly Criterion sizing.
        """
        win_size = self.window_size
        min_p = min(win_size, 100, len(probabilities))
        rolling_rank = probabilities.rolling(window=win_size, min_periods=min_p).rank(pct=True)
        # Sigmoid smoothing for stable kelly inputs
        return 1 / (1 + np.exp(-10 * (rolling_rank - 0.5)))

    def get_z_score(self, probabilities: pd.Series) -> pd.Series:
        """
        Calculate rolling z-score of probabilities.

        Useful for regime detection or alternative calibration strategies.
        Z-Score = (Value - Mean) / StdDev

        Args:
            probabilities: Series of raw probability values.

        Returns:
            pd.Series: Z-scores indicating how many standard deviations
                      the current value is from the moving average.
        """
        if probabilities.empty:
            return pd.Series(dtype=float)

        min_p = min(self.window_size, 100, len(probabilities))

        rolling_mean = probabilities.rolling(window=self.window_size, min_periods=min_p).mean()

        rolling_std = probabilities.rolling(window=self.window_size, min_periods=min_p).std()

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)

        z_score = (probabilities - rolling_mean) / rolling_std
        return z_score.fillna(0.0)
