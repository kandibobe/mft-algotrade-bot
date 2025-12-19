#!/usr/bin/env python3
"""
Data Inspector - просмотр и анализ скачанных данных
====================================================

Показывает статистику по скачанным историческим данным.

Usage:
    python scripts/inspect_data.py
    python scripts/inspect_data.py --pair BTC/USDT --timeframe 5m
"""

import argparse
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List


class DataInspector:
    """Инспектор для анализа скачанных данных."""

    def __init__(self, data_dir: str = "user_data/data/binance"):
        self.data_dir = Path(data_dir)

    def list_available_data(self) -> Dict[str, List[str]]:
        """
        Получить список всех доступных данных.

        Returns:
            Dict с парами и доступными таймфреймами
        """
        data = {}

        if not self.data_dir.exists():
            print(f"❌ Data directory not found: {self.data_dir}")
            return data

        # Найти все JSON файлы
        for file in self.data_dir.glob("*.json"):
            # Формат: BTC_USDT-5m.json
            name = file.stem

            if '-' in name:
                pair_str, timeframe = name.rsplit('-', 1)
                pair = pair_str.replace('_', '/')

                if pair not in data:
                    data[pair] = []
                data[pair].append(timeframe)

        return data

    def inspect_pair(self, pair: str, timeframe: str = "5m") -> None:
        """
        Инспектировать данные конкретной пары.

        Args:
            pair: Пара (например, BTC/USDT)
            timeframe: Таймфрейм (например, 5m)
        """
        # Преобразовать пару в имя файла
        pair_filename = pair.replace('/', '_')
        filename = f"{pair_filename}-{timeframe}.json"
        filepath = self.data_dir / filename

        if not filepath.exists():
            print(f"❌ File not found: {filepath}")
            print(f"\nAvailable data:")
            self.show_available_data()
            return

        # Загрузить данные
        with open(filepath, 'r') as f:
            data = json.load(f)

        if not data:
            print(f"❌ No data in file: {filepath}")
            return

        # Конвертировать в DataFrame
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Статистика
        print("\n" + "="*70)
        print(f"📊 DATA INSPECTION: {pair} ({timeframe})")
        print("="*70)

        print(f"\n📅 Time Range:")
        print(f"   Start:    {df.index[0]}")
        print(f"   End:      {df.index[-1]}")
        print(f"   Duration: {(df.index[-1] - df.index[0]).days} days")
        print(f"   Candles:  {len(df):,}")

        print(f"\n💰 Price Statistics:")
        print(f"   Current:  ${df['close'].iloc[-1]:,.2f}")
        print(f"   High:     ${df['high'].max():,.2f}")
        print(f"   Low:      ${df['low'].min():,.2f}")
        print(f"   Avg:      ${df['close'].mean():,.2f}")
        print(f"   Std Dev:  ${df['close'].std():,.2f}")

        # Returns
        returns = df['close'].pct_change().dropna()
        print(f"\n📈 Returns:")
        print(f"   Total:    {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:+.2f}%")
        print(f"   Daily Avg:{returns.mean() * 100:.4f}%")
        print(f"   Volatility:{returns.std() * 100:.2f}%")
        print(f"   Max Gain: {returns.max() * 100:+.2f}%")
        print(f"   Max Loss: {returns.min() * 100:+.2f}%")

        print(f"\n📊 Volume:")
        print(f"   Total:    {df['volume'].sum():,.0f} {pair.split('/')[0]}")
        print(f"   Avg:      {df['volume'].mean():,.0f}")
        print(f"   Max:      {df['volume'].max():,.0f}")

        # Data quality
        print(f"\n✅ Data Quality:")
        missing = df.isnull().sum().sum()
        print(f"   Missing values: {missing}")

        # Check for gaps
        expected_interval = pd.Timedelta(timeframe)
        actual_intervals = df.index.to_series().diff().dropna()
        gaps = actual_intervals[actual_intervals > expected_interval * 1.5]

        if len(gaps) > 0:
            print(f"   ⚠️  Time gaps: {len(gaps)} (max: {gaps.max()})")
        else:
            print(f"   Time gaps: None ✅")

        # Recent candles
        print(f"\n🕐 Recent Candles (last 5):")
        print(df[['open', 'high', 'low', 'close', 'volume']].tail(5).to_string())

        print("\n" + "="*70)
        print(f"✅ Data inspection complete!")
        print(f"📁 File: {filepath}")
        print("="*70 + "\n")

    def show_available_data(self) -> None:
        """Показать все доступные данные."""
        data = self.list_available_data()

        if not data:
            print("\n❌ No data found!")
            print(f"Download data first:")
            print(f"  docker exec stoic_freqtrade freqtrade download-data \\")
            print(f"    --exchange binance \\")
            print(f"    --timeframe 5m \\")
            print(f"    --pairs BTC/USDT ETH/USDT \\")
            print(f"    --days 30")
            return

        print("\n" + "="*70)
        print("📦 AVAILABLE DATA")
        print("="*70)

        for pair, timeframes in sorted(data.items()):
            print(f"\n{pair}")
            for tf in sorted(timeframes):
                pair_filename = pair.replace('/', '_')
                filename = f"{pair_filename}-{tf}.json"
                filepath = self.data_dir / filename

                # Get file size
                size_mb = filepath.stat().st_size / (1024 * 1024)

                # Count candles
                with open(filepath, 'r') as f:
                    candles = len(json.load(f))

                print(f"  - {tf:5s} | {candles:6,} candles | {size_mb:6.2f} MB")

        print("\n" + "="*70 + "\n")

    def compare_pairs(self, pairs: List[str], timeframe: str = "5m") -> None:
        """
        Сравнить несколько пар.

        Args:
            pairs: Список пар для сравнения
            timeframe: Таймфрейм
        """
        print("\n" + "="*70)
        print(f"📊 PAIRS COMPARISON ({timeframe})")
        print("="*70)

        comparison = []

        for pair in pairs:
            pair_filename = pair.replace('/', '_')
            filename = f"{pair_filename}-{timeframe}.json"
            filepath = self.data_dir / filename

            if not filepath.exists():
                print(f"⚠️  {pair}: No data")
                continue

            with open(filepath, 'r') as f:
                data = json.load(f)

            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['close'] = df['close'].astype(float)

            # Статистика
            total_return = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100
            returns = pd.Series(df['close']).pct_change().dropna()
            volatility = returns.std() * 100

            comparison.append({
                'Pair': pair,
                'Candles': len(df),
                'Price': f"${float(df['close'].iloc[-1]):,.2f}",
                'Return %': f"{total_return:+.2f}%",
                'Volatility %': f"{volatility:.2f}%",
                'Avg Volume': f"{df['volume'].mean():,.0f}"
            })

        if comparison:
            comp_df = pd.DataFrame(comparison)
            print(f"\n{comp_df.to_string(index=False)}")

        print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect downloaded trading data"
    )
    parser.add_argument(
        '--pair',
        type=str,
        help='Pair to inspect (e.g., BTC/USDT)'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        default='5m',
        help='Timeframe (default: 5m)'
    )
    parser.add_argument(
        '--compare',
        type=str,
        nargs='+',
        help='Compare multiple pairs (e.g., --compare BTC/USDT ETH/USDT)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='user_data/data/binance',
        help='Data directory'
    )

    args = parser.parse_args()

    inspector = DataInspector(data_dir=args.data_dir)

    if args.compare:
        inspector.compare_pairs(args.compare, args.timeframe)
    elif args.pair:
        inspector.inspect_pair(args.pair, args.timeframe)
    else:
        inspector.show_available_data()


if __name__ == "__main__":
    main()
