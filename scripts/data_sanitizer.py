#!/usr/bin/env python3
"""
Data Sanitizer for MFT Algo Trading Bot - Phase 0

This script performs data hygiene on OHLCV data to ensure we have
365 days of clean 5-minute data for BTC/USDT and ETH/USDT.

Key operations:
1. Load data from user_data/data/binance/
2. Check for gaps > 5 minutes
3. Forward-fill small gaps (≤ 15 minutes)
4. Drop rows with NaN/Inf values
5. Save cleaned data to user_data/data/cleaned/

Usage:
    python scripts/data_sanitizer.py [--symbols BTC/USDT ETH/USDT] [--timeframe 5m]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

# Add project root to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import get_ohlcv, save_to_parquet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT"]
DEFAULT_TIMEFRAME = "5m"
DATA_DIR = Path("user_data/data")
CLEANED_DIR = DATA_DIR / "cleaned"
MAX_ALLOWED_GAP_MINUTES = 5  # Gaps larger than this are considered problematic
MAX_FORWARD_FILL_MINUTES = 15  # Maximum gap size to forward-fill


def load_raw_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Load raw OHLCV data for a symbol and timeframe.
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Candle timeframe (e.g., '5m')
        
    Returns:
        DataFrame with OHLCV data indexed by datetime
    """
    try:
        logger.info(f"Loading data for {symbol} {timeframe}")
        df = get_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            start=None,
            end=None,
            exchange="binance",
            data_dir=DATA_DIR,
            use_cache=False  # Don't use cache for data cleaning
        )
        
        # Ensure we have a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            else:
                df.index = pd.to_datetime(df.index)
        
        logger.info(f"Loaded {len(df)} rows for {symbol} {timeframe}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        
        return df
    
    except Exception as e:
        logger.error(f"Failed to load data for {symbol} {timeframe}: {e}")
        raise


def analyze_gaps(df: pd.DataFrame, timeframe: str) -> Dict:
    """
    Analyze gaps in the time series data.
    
    Args:
        df: DataFrame with DatetimeIndex
        timeframe: Candle timeframe (e.g., '5m')
        
    Returns:
        Dictionary with gap analysis results
    """
    if len(df) < 2:
        return {
            "total_gaps": 0,
            "large_gaps": 0,
            "max_gap_minutes": 0,
            "avg_gap_minutes": 0,
            "gap_details": []
        }
    
    # Calculate expected frequency in minutes
    timeframe_to_minutes = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '1h': 60,
        '4h': 240,
        '1d': 1440
    }
    
    expected_minutes = timeframe_to_minutes.get(timeframe, 5)
    
    # Calculate time differences between consecutive rows
    time_diffs = df.index.to_series().diff().dt.total_seconds() / 60  # Convert to minutes
    
    # Identify gaps (differences larger than expected)
    gaps = time_diffs[time_diffs > expected_minutes * 1.1]  # 10% tolerance
    
    gap_details = []
    for idx, gap_minutes in gaps.items():
        gap_details.append({
            "gap_start": df.index[df.index.get_loc(idx) - 1],
            "gap_end": idx,
            "gap_minutes": gap_minutes,
            "expected_minutes": expected_minutes,
            "is_large": gap_minutes > MAX_ALLOWED_GAP_MINUTES
        })
    
    result = {
        "total_gaps": len(gaps),
        "large_gaps": sum(1 for g in gap_details if g["is_large"]),
        "max_gap_minutes": gaps.max() if len(gaps) > 0 else 0,
        "avg_gap_minutes": gaps.mean() if len(gaps) > 0 else 0,
        "gap_details": gap_details
    }
    
    return result


def check_data_quality(df: pd.DataFrame) -> Dict:
    """
    Check data quality issues.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dictionary with quality metrics
    """
    # Check for NaN/Inf values
    nan_count = df.isna().sum().sum()
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    
    # Check for zero or negative values where they shouldn't be
    zero_volume = (df['volume'] <= 0).sum()
    zero_close = (df['close'] <= 0).sum()
    
    # Check for outliers (prices more than 3 standard deviations from mean)
    price_columns = ['open', 'high', 'low', 'close']
    outlier_counts = {}
    for col in price_columns:
        if col in df.columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outlier_counts[col] = (z_scores > 3).sum()
    
    return {
        "total_rows": len(df),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "zero_volume_count": zero_volume,
        "zero_close_count": zero_close,
        "outlier_counts": outlier_counts,
        "date_range_days": (df.index.max() - df.index.min()).days,
        "candles_per_day": len(df) / max((df.index.max() - df.index.min()).days, 1)
    }


def clean_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Clean the data by:
    1. Forward-filling small gaps (≤ 15 minutes)
    2. Dropping rows with NaN/Inf values
    3. Ensuring proper datetime index
    
    Args:
        df: Raw DataFrame
        timeframe: Candle timeframe
        
    Returns:
        Cleaned DataFrame
    """
    logger.info(f"Starting data cleaning for {len(df)} rows")
    
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # 1. Ensure proper datetime index
    if not isinstance(df_clean.index, pd.DatetimeIndex):
        if 'date' in df_clean.columns:
            df_clean['date'] = pd.to_datetime(df_clean['date'])
            df_clean.set_index('date', inplace=True)
        else:
            df_clean.index = pd.to_datetime(df_clean.index)
    
    # 2. Sort by index
    df_clean = df_clean.sort_index()
    
    # 3. Calculate expected frequency
    timeframe_to_minutes = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}
    expected_minutes = timeframe_to_minutes.get(timeframe, 5)
    
    # 4. Resample to regular frequency and forward-fill small gaps
    # Create a regular date range at the expected frequency
    full_range = pd.date_range(
        start=df_clean.index.min(),
        end=df_clean.index.max(),
        freq=f'{expected_minutes}min'
    )
    
    # Reindex to the full range - this creates NaN for missing timestamps
    df_clean = df_clean.reindex(full_range)
    
    # 5. Identify gaps that can be forward-filled (≤ 15 minutes)
    # Count consecutive NaN rows
    nan_mask = df_clean['close'].isna() if 'close' in df_clean.columns else pd.Series(False, index=df_clean.index)
    
    if nan_mask.any():
        # Find groups of consecutive NaNs
        nan_groups = (nan_mask != nan_mask.shift()).cumsum()
        group_sizes = nan_mask.groupby(nan_groups).transform('sum')
        
        # Only forward-fill gaps of size ≤ MAX_FORWARD_FILL_MINUTES/expected_minutes candles
        max_fill_candles = MAX_FORWARD_FILL_MINUTES // expected_minutes
        fill_mask = group_sizes <= max_fill_candles
        
        # Forward fill small gaps
        df_clean = df_clean.ffill(limit=max_fill_candles)
        
        # Count how many values were filled
        filled_count = fill_mask.sum()
        logger.info(f"Forward-filled {filled_count} small gaps (≤ {MAX_FORWARD_FILL_MINUTES} minutes)")
    
    # 6. Drop rows that still have NaN values (large gaps or missing data)
    rows_before = len(df_clean)
    df_clean = df_clean.dropna()
    rows_after = len(df_clean)
    rows_dropped = rows_before - rows_after
    
    if rows_dropped > 0:
        logger.warning(f"Dropped {rows_dropped} rows with NaN values")
    
    # 7. Remove infinite values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    inf_mask = np.isinf(df_clean[numeric_cols]).any(axis=1)
    if inf_mask.any():
        df_clean = df_clean[~inf_mask]
        logger.warning(f"Removed {inf_mask.sum()} rows with infinite values")
    
    # 8. Ensure all numeric values are positive where expected
    for col in ['open', 'high', 'low', 'close']:
        if col in df_clean.columns:
            # Replace non-positive values with small positive number
            non_positive_mask = df_clean[col] <= 0
            if non_positive_mask.any():
                df_clean.loc[non_positive_mask, col] = 0.01
                logger.warning(f"Fixed {non_positive_mask.sum()} non-positive values in {col}")
    
    logger.info(f"Cleaning complete: {rows_before} -> {rows_after} rows ({rows_dropped} dropped)")
    
    return df_clean


def save_cleaned_data(df: pd.DataFrame, symbol: str, timeframe: str, output_dir: Path) -> Path:
    """
    Save cleaned data to Parquet format.
    
    Args:
        df: Cleaned DataFrame
        symbol: Trading pair
        timeframe: Candle timeframe
        output_dir: Directory to save cleaned data
        
    Returns:
        Path to saved file
    """
    # Create cleaned directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Normalize symbol for filename
    symbol_normalized = symbol.replace("/", "_")
    
    # Save to Parquet
    output_path = output_dir / f"{symbol_normalized}-{timeframe}.parquet"
    save_to_parquet(df, output_path)
    
    logger.info(f"Saved cleaned data to {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return output_path


def generate_report(symbol: str, timeframe: str, raw_df: pd.DataFrame, 
                   clean_df: pd.DataFrame, gap_analysis: Dict, 
                   quality_before: Dict, quality_after: Dict) -> Dict:
    """
    Generate a comprehensive report for the data cleaning process.
    
    Returns:
        Dictionary with report data
    """
    report = {
        "symbol": symbol,
        "timeframe": timeframe,
        "processing_time": pd.Timestamp.now().isoformat(),
        "raw_data": {
            "rows": len(raw_df),
            "date_range": {
                "start": raw_df.index.min().isoformat(),
                "end": raw_df.index.max().isoformat()
            },
            "days": (raw_df.index.max() - raw_df.index.min()).days,
            "quality": quality_before
        },
        "cleaned_data": {
            "rows": len(clean_df),
            "date_range": {
                "start": clean_df.index.min().isoformat(),
                "end": clean_df.index.max().isoformat()
            },
            "days": (clean_df.index.max() - clean_df.index.min()).days,
            "quality": quality_after
        },
        "gap_analysis": gap_analysis,
        "cleaning_stats": {
            "rows_dropped": len(raw_df) - len(clean_df),
            "percent_kept": (len(clean_df) / len(raw_df) * 100) if len(raw_df) > 0 else 0,
            "large_gaps_found": gap_analysis["large_gaps"],
            "small_gaps_filled": gap_analysis["total_gaps"] - gap_analysis["large_gaps"]
        },
        "success_criteria": {
            "has_365_days": (clean_df.index.max() - clean_df.index.min()).days >= 365,
            "no_nan_inf": quality_after["nan_count"] == 0 and quality_after["inf_count"] == 0,
            "max_gap_ok": gap_analysis["max_gap_minutes"] <= MAX_ALLOWED_GAP_MINUTES
        }
    }
    
    return report


def print_report(report: Dict):
    """Print a human-readable report."""
    print("\n" + "="*80)
    print(f"DATA SANITIZATION REPORT: {report['symbol']} {report['timeframe']}")
    print("="*80)
    
    print(f"\n📊 SUMMARY:")
    print(f"  • Symbol: {report['symbol']}")
    print(f"  • Timeframe: {report['timeframe']}")
    print(f"  • Processing time: {report['processing_time']}")
    
    print(f"\n📈 DATA STATISTICS:")
    print(f"  • Raw data: {report['raw_data']['rows']:,} rows "
          f"({report['raw_data']['days']} days)")
    print(f"  • Cleaned data: {report['cleaned_data']['rows']:,} rows "
          f"({report['cleaned_data']['days']} days)")
    print(f"  • Rows kept: {report['cleaning_stats']['percent_kept']:.1f}%")
    
    print(f"\n🔍 GAP ANALYSIS:")
    print(f"  • Total gaps found: {report['gap_analysis']['total_gaps']}")
    print(f"  • Large gaps (> {MAX_ALLOWED_GAP_MINUTES} min): {report['gap_analysis']['large_gaps']}")
    print(f"  • Maximum gap: {report['gap_analysis']['max_gap_minutes']:.1f} minutes")
    
    print(f"\n🧹 CLEANING RESULTS:")
    print(f"  • NaN values removed: {report['raw_data']['quality']['nan_count']} → "
          f"{report['cleaned_data']['quality']['nan_count']}")
    print(f"  • Inf values removed: {report['raw_data']['quality']['inf_count']} → "
          f"{report['cleaned_data']['quality']['inf_count']}")
    
    print(f"\n✅ SUCCESS CRITERIA:")
    criteria = report['success_criteria']
    print(f"  • Has 365+ days of data: {'✓' if criteria['has_365_days'] else '✗'} "
          f"({report['cleaned_data']['days']} days)")
    print(f"  • No NaN/Inf values: {'✓' if criteria['no_nan_inf'] else '✗'}")
    print(f"  • Max gap ≤ {MAX_ALLOWED_GAP_MINUTES} min: {'✓' if criteria['max_gap_ok'] else '✗'} "
          f"({report['gap_analysis']['max_gap_minutes']:.1f} min)")
    
    # Check if all criteria are met
    all_met = all(criteria.values())
    print(f"\n🎯 OVERALL STATUS: {'PASS' if all_met else 'FAIL'}")
    
    if not criteria['has_365_days']:
        print(f"   ⚠️  Need {365 - report['cleaned_data']['days']} more days of data")
    if not criteria['no_nan_inf']:
        print(f"   ⚠️  Data still contains NaN/Inf values")
    if not criteria['max_gap_ok']:
        print(f"   ⚠️  Large gaps present in data")
    
    print("="*80 + "\n")


def process_symbol(symbol: str, timeframe: str, output_dir: Path) -> bool:
    """
    Process a single symbol/timeframe combination.
    
    Returns:
        True if all success criteria are met, False otherwise
    """
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {symbol} {timeframe}")
        logger.info(f"{'='*60}")
        
        # 1. Load raw data
        raw_df = load_raw_data(symbol, timeframe)
        
        # 2. Analyze gaps in raw data
        gap_analysis = analyze_gaps(raw_df, timeframe)
        
        # 3. Check raw data quality
        quality_before = check_data_quality(raw_df)
        
        # 4. Clean the data
        clean_df = clean_data(raw_df, timeframe)
        
        # 5. Check cleaned data quality
        quality_after = check_data_quality(clean_df)
        
        # 6. Save cleaned data
        output_path = save_cleaned_data(clean_df, symbol, timeframe, output_dir)
        
        # 7. Generate and print report
        report = generate_report(
            symbol, timeframe, raw_df, clean_df, 
            gap_analysis, quality_before, quality_after
        )
        print_report(report)
        
        # 8. Return whether all success criteria are met
        return all(report["success_criteria"].values())
        
    except Exception as e:
        logger.error(f"Failed to process {symbol} {timeframe}: {e}")
        return False


def main():
    """Main entry point for the data sanitizer."""
    parser = argparse.ArgumentParser(
        description="Clean and sanitize OHLCV data for ML training"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help=f"Trading symbols to process (default: {DEFAULT_SYMBOLS})"
    )
    parser.add_argument(
        "--timeframe",
        default=DEFAULT_TIMEFRAME,
        help=f"Timeframe to process (default: {DEFAULT_TIMEFRAME})"
    )
    parser.add_argument(
        "--output-dir",
        default=str(CLEANED_DIR),
        help=f"Output directory for cleaned data (default: {CLEANED_DIR})"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting Data Sanitizer - Phase 0")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Update output directory if specified
    output_dir = Path(args.output_dir)
    
    # Process each symbol
    results = []
    for symbol in args.symbols:
        success = process_symbol(symbol, args.timeframe, output_dir)
        results.append((symbol, success))
    
    # Print summary
    print("\n" + "="*80)
    print("DATA SANITIZATION SUMMARY")
    print("="*80)
    
    all_passed = True
    for symbol, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {symbol}: {status}")
        if not success:
            all_passed = False
    
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("="*80)
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
