"""
Tests for Probability Calibration Module.
"""

import numpy as np
import pandas as pd

from src.ml.calibration import ProbabilityCalibrator


def test_calibration_init():
    """Test initialization of ProbabilityCalibrator."""
    calibrator = ProbabilityCalibrator(window_size=100, percentile_threshold=0.85)
    assert calibrator.window_size == 100
    assert calibrator.percentile_threshold == 0.85

def test_is_signal_empty():
    """Test behavior with empty input."""
    calibrator = ProbabilityCalibrator()
    empty_series = pd.Series([], dtype=float)
    result = calibrator.is_signal(empty_series)
    assert result.empty
    assert len(result) == 0

def test_is_signal_basic():
    """Test basic signal generation logic."""
    calibrator = ProbabilityCalibrator(window_size=10, percentile_threshold=0.80)

    # Create a series where the last element is the maximum
    probs = pd.Series(np.linspace(0.1, 0.9, 10))
    # [0.1, 0.188..., ..., 0.9]
    # The last value 0.9 is the max in the window of 10

    signals = calibrator.is_signal(probs)

    # The last value should be a signal because it's the 100th percentile (rank 1.0)
    # 1.0 > 0.80 -> True
    assert signals.iloc[-1] == True

    # The first value has no history, rank is 1.0?
    # Pandas rank with min_periods might behave differently.
    # We rely on our min_periods logic.

def test_is_signal_spike():
    """Test detection of probability spikes."""
    calibrator = ProbabilityCalibrator(window_size=20, percentile_threshold=0.90)

    # 100 random values around 0.3
    np.random.seed(42)
    probs = pd.Series(np.random.normal(0.3, 0.05, 100))

    # Add a huge spike
    probs.iloc[50] = 0.9

    signals = calibrator.is_signal(probs)

    # The spike should be detected
    assert signals.iloc[50] == True

    # Normal values should generally be False (approx 90% of them)
    # We can't assert exact count due to randomness but ratio should be low
    assert signals.mean() < 0.2

def test_min_periods_logic():
    """Test that min_periods logic handles short series gracefully."""
    # Window size 100, but data length 10
    calibrator = ProbabilityCalibrator(window_size=100, percentile_threshold=0.90)

    probs = pd.Series(np.linspace(0.1, 0.5, 10))
    # Add spike
    probs.iloc[-1] = 0.9

    signals = calibrator.is_signal(probs)

    # Should not crash and should detect the spike because min_periods adapts
    assert len(signals) == 10
    assert signals.iloc[-1] == True

def test_z_score():
    """Test z-score calculation."""
    calibrator = ProbabilityCalibrator(window_size=10)

    # Constant series -> std=0 -> z-score should handle div by zero (fillna 0 or similar)
    probs = pd.Series([0.5] * 20)
    z_scores = calibrator.get_z_score(probs)

    # Should be 0 (mean=0.5, val=0.5)
    assert (z_scores == 0).all()

    # Variable series
    probs = pd.Series([0.1, 0.2, 0.1, 0.2, 0.9])
    z_scores = calibrator.get_z_score(probs)

    # The last value is far from mean, should have high z-score
    assert z_scores.iloc[-1] > 1.0
