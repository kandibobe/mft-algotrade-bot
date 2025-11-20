#!/usr/bin/env python3
"""
Stoic Citadel - Data Verification Script
=========================================

Checks downloaded data for:
- Missing candles (gaps)
- Anomalies (price spikes)
- Data quality issues

Usage:
    python scripts/verify_data.py
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


class DataVerifier:
    """Verify quality of downloaded trading data."""

    def __init__(self, data_dir: str = "./user_data/data/binance"):
        self.data_dir = Path(data_dir)
        self.issues: List[Dict] = []

    def check_gaps(
        self, df: pd.DataFrame, pair: str, timeframe: str
    ) -> List[Dict]:
        """Check for missing candles (gaps in data)."""
        gaps = []

        # Expected frequency based on timeframe
        freq_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "1h": "1H",
            "4h": "4H",
            "1d": "1D",
        }
        expected_freq = freq_map.get(timeframe, "5min")

        # Create expected datetime range
        expected_range = pd.date_range(
            start=df.index[0], end=df.index[-1], freq=expected_freq
        )

        # Find missing timestamps
        missing = expected_range.difference(df.index)

        if len(missing) > 0:
            gaps.append(
                {
                    "pair": pair,
                    "timeframe": timeframe,
                    "issue": "gaps",
                    "count": len(missing),
                    "percentage": (len(missing) / len(expected_range)) * 100,
                    "details": f"Missing {len(missing)} candles out of {len(expected_range)}",
                }
            )

        return gaps

    def check_anomalies(
        self, df: pd.DataFrame, pair: str, timeframe: str
    ) -> List[Dict]:
        """Check for price anomalies (spikes, invalid values)."""
        anomalies = []

        # Check for zero or negative prices
        invalid_prices = (
            (df["open"] <= 0)
            | (df["high"] <= 0)
            | (df["low"] <= 0)
            | (df["close"] <= 0)
        )
        if invalid_prices.any():
            anomalies.append(
                {
                    "pair": pair,
                    "timeframe": timeframe,
                    "issue": "invalid_prices",
                    "count": invalid_prices.sum(),
                    "details": "Found zero or negative prices",
                }
            )

        # Check for illogical OHLC (high < low, etc.)
        illogical = (df["high"] < df["low"]) | (df["high"] < df["open"]) | (
            df["high"] < df["close"]
        )
        if illogical.any():
            anomalies.append(
                {
                    "pair": pair,
                    "timeframe": timeframe,
                    "issue": "illogical_ohlc",
                    "count": illogical.sum(),
                    "details": "OHLC values don't make sense",
                }
            )

        # Check for extreme price movements (>50% in one candle)
        price_change = df["close"].pct_change().abs()
        extreme_moves = price_change > 0.5
        if extreme_moves.any():
            anomalies.append(
                {
                    "pair": pair,
                    "timeframe": timeframe,
                    "issue": "extreme_moves",
                    "count": extreme_moves.sum(),
                    "details": f"Found {extreme_moves.sum()} candles with >50% price change",
                }
            )

        # Check for zero volume
        zero_volume = df["volume"] == 0
        if zero_volume.any():
            anomalies.append(
                {
                    "pair": pair,
                    "timeframe": timeframe,
                    "issue": "zero_volume",
                    "count": zero_volume.sum(),
                    "percentage": (zero_volume.sum() / len(df)) * 100,
                    "details": f"{zero_volume.sum()} candles with zero volume",
                }
            )

        return anomalies

    def load_data(self, file_path: Path) -> pd.DataFrame:
        """Load OHLCV data from JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)

        df = pd.DataFrame(
            data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        return df

    def verify_all(self) -> Tuple[int, int]:
        """Verify all data files in the data directory."""
        if not self.data_dir.exists():
            print(f"❌ Data directory not found: {self.data_dir}")
            return 0, 0

        # Find all JSON files
        json_files = list(self.data_dir.glob("*.json"))

        if not json_files:
            print(f"❌ No data files found in {self.data_dir}")
            return 0, 0

        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║         STOIC CITADEL - DATA VERIFICATION                     ║")
        print("╚═══════════════════════════════════════════════════════════════╝")
        print()
        print(f"📂 Data directory: {self.data_dir}")
        print(f"🔍 Files found: {len(json_files)}")
        print()

        total_files = 0
        clean_files = 0

        for file_path in json_files:
            # Parse filename (e.g., BTC_USDT-5m.json)
            filename = file_path.stem
            parts = filename.split("-")
            pair = parts[0].replace("_", "/")
            timeframe = parts[1] if len(parts) > 1 else "unknown"

            print(f"Checking {pair} ({timeframe})... ", end="")

            try:
                df = self.load_data(file_path)

                # Run checks
                gaps = self.check_gaps(df, pair, timeframe)
                anomalies = self.check_anomalies(df, pair, timeframe)

                issues = gaps + anomalies
                self.issues.extend(issues)

                total_files += 1

                if not issues:
                    print("✅ OK")
                    clean_files += 1
                else:
                    print(f"⚠️  {len(issues)} issues found")
                    for issue in issues:
                        print(f"   - {issue['issue']}: {issue['details']}")

            except Exception as e:
                print(f"❌ Error: {e}")
                self.issues.append(
                    {
                        "pair": pair,
                        "timeframe": timeframe,
                        "issue": "load_error",
                        "details": str(e),
                    }
                )

        print()
        print("═" * 65)
        print("SUMMARY")
        print("═" * 65)
        print(f"Total files checked: {total_files}")
        print(f"Clean files: {clean_files}")
        print(f"Files with issues: {total_files - clean_files}")
        print(f"Total issues: {len(self.issues)}")
        print()

        if self.issues:
            print("⚠️  Issues found:")
            issue_types = {}
            for issue in self.issues:
                issue_type = issue["issue"]
                issue_types[issue_type] = issue_types.get(issue_type, 0) + 1

            for issue_type, count in issue_types.items():
                print(f"  - {issue_type}: {count}")
        else:
            print("✅ All data is clean!")

        print()

        return total_files, clean_files


def main():
    """Main execution."""
    verifier = DataVerifier()
    total, clean = verifier.verify_all()

    if total == 0:
        print("💡 Run ./scripts/download_data.sh first to download data")
        return 1

    if clean == total:
        print("✅ Data verification passed!")
        return 0
    else:
        print("⚠️  Some data quality issues detected.")
        print("💡 Consider re-downloading data for affected pairs.")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
