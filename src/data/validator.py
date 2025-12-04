"""
Stoic Citadel - Data Validator
===============================

Ensures data integrity and quality for reliable backtesting.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def validate_ohlcv(df: pd.DataFrame, strict: bool = False) -> Tuple[bool, List[str]]:
    """
    Validate OHLCV data for common issues.
    
    Checks:
    - Required columns present
    - No missing values (NaN)
    - Price integrity (high >= low, open/close within high-low)
    - No negative values
    - No duplicate timestamps
    - Chronological order
    - No suspicious gaps
    
    Args:
        df: DataFrame with OHLCV data
        strict: If True, any issue fails validation
        
    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []
    
    # Check 1: Required columns
    required = {'open', 'high', 'low', 'close', 'volume'}
    cols = set(df.columns.str.lower())
    missing = required - cols
    if missing:
        issues.append(f"Missing columns: {missing}")
        return False, issues  # Can't continue without columns
    
    # Standardize column names for checks
    df_check = df.copy()
    df_check.columns = df_check.columns.str.lower()
    
    # Check 2: Missing values
    nan_counts = df_check[['open', 'high', 'low', 'close', 'volume']].isna().sum()
    nan_total = nan_counts.sum()
    if nan_total > 0:
        issues.append(f"Missing values: {nan_counts.to_dict()}")
    
    # Check 3: Price integrity
    price_errors = (
        (df_check['high'] < df_check['low']) |
        (df_check['open'] > df_check['high']) |
        (df_check['open'] < df_check['low']) |
        (df_check['close'] > df_check['high']) |
        (df_check['close'] < df_check['low'])
    )
    if price_errors.any():
        n_errors = price_errors.sum()
        issues.append(f"Price integrity errors: {n_errors} candles")
    
    # Check 4: Negative values
    negative = (
        (df_check['open'] < 0) |
        (df_check['high'] < 0) |
        (df_check['low'] < 0) |
        (df_check['close'] < 0) |
        (df_check['volume'] < 0)
    )
    if negative.any():
        issues.append(f"Negative values found: {negative.sum()} rows")
    
    # Check 5: Duplicate timestamps (if datetime index)
    if isinstance(df_check.index, pd.DatetimeIndex):
        duplicates = df_check.index.duplicated()
        if duplicates.any():
            issues.append(f"Duplicate timestamps: {duplicates.sum()}")
        
        # Check 6: Chronological order
        if not df_check.index.is_monotonic_increasing:
            issues.append("Data not in chronological order")
    
    # Check 7: Zero volume (suspicious)
    zero_vol_pct = (df_check['volume'] == 0).mean() * 100
    if zero_vol_pct > 5:
        issues.append(f"High zero-volume candles: {zero_vol_pct:.1f}%")
    
    # Determine validity
    if strict:
        is_valid = len(issues) == 0
    else:
        # Allow minor issues (warnings) but fail on critical ones
        critical_keywords = ['Missing columns', 'Negative values', 'Price integrity']
        is_valid = not any(
            any(kw in issue for kw in critical_keywords)
            for issue in issues
        )
    
    if issues:
        for issue in issues:
            logger.warning(f"Data validation: {issue}")
    else:
        logger.info("Data validation passed")
    
    return is_valid, issues


def check_data_integrity(
    df: pd.DataFrame,
    expected_timeframe: str = '5m'
) -> Tuple[bool, dict]:
    """
    Check data completeness and detect gaps.
    
    Returns:
        Tuple of (has_gaps, gap_info_dict)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return False, {'error': 'Not a datetime index'}
    
    # Calculate expected frequency
    tf_map = {
        '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
        '1h': '1h', '4h': '4h', '1d': '1D'
    }
    freq = tf_map.get(expected_timeframe, '5min')
    
    # Generate expected index
    expected = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=freq
    )
    
    # Find missing timestamps
    missing = expected.difference(df.index)
    
    gap_info = {
        'expected_candles': len(expected),
        'actual_candles': len(df),
        'missing_candles': len(missing),
        'completeness_pct': (len(df) / len(expected)) * 100 if len(expected) > 0 else 100,
        'largest_gap': None
    }
    
    # Find largest gap
    if len(missing) > 0:
        # Group consecutive missing timestamps
        missing_series = pd.Series(missing)
        time_diffs = missing_series.diff()
        gap_starts = missing_series[time_diffs != pd.Timedelta(freq)].tolist()
        
        if gap_starts:
            gap_info['largest_gap'] = str(max(
                (missing_series.iloc[i+1] - missing_series.iloc[i] 
                 for i in range(len(missing_series)-1)),
                default=pd.Timedelta(0)
            ))
    
    has_gaps = len(missing) > 0
    
    if has_gaps:
        logger.warning(
            f"Data gaps detected: {gap_info['missing_candles']} missing candles "
            f"({gap_info['completeness_pct']:.1f}% complete)"
        )
    else:
        logger.info(f"Data integrity check passed: 100% complete")
    
    return has_gaps, gap_info


def fill_gaps(
    df: pd.DataFrame,
    method: str = 'ffill',
    timeframe: str = '5m'
) -> pd.DataFrame:
    """
    Fill gaps in OHLCV data.
    
    Methods:
    - ffill: Forward fill (use last known values)
    - bfill: Backward fill
    - interpolate: Linear interpolation
    """
    tf_map = {
        '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
        '1h': '1h', '4h': '4h', '1d': '1D'
    }
    freq = tf_map.get(timeframe, '5min')
    
    # Reindex with complete datetime range
    complete_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=freq
    )
    
    df_filled = df.reindex(complete_index)
    
    if method == 'ffill':
        df_filled = df_filled.ffill()
    elif method == 'bfill':
        df_filled = df_filled.bfill()
    elif method == 'interpolate':
        df_filled = df_filled.interpolate(method='linear')
    
    # Volume should be 0 for filled candles, not interpolated
    if 'volume' in df_filled.columns:
        df_filled.loc[~df_filled.index.isin(df.index), 'volume'] = 0
    
    return df_filled
