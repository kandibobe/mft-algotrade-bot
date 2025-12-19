#!/usr/bin/env python3
"""
Quick Backtest Runner
=====================

Удобный способ запуска бэктестов с готовыми профилями.

Usage:
    # Quick test (последние 7 дней)
    python scripts/run_backtest.py --profile quick

    # Full test (30 дней)
    python scripts/run_backtest.py --profile full

    # Custom
    python scripts/run_backtest.py --pair BTC/USDT --days 14
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta


class BacktestProfile:
    """Профили для быстрого запуска бэктестов."""

    PROFILES = {
        'quick': {
            'description': 'Quick test - последние 7 дней',
            'pairs': ['BTC/USDT'],
            'days': 7,
            'timeframe': '5m'
        },
        'full': {
            'description': 'Full test - последние 30 дней',
            'pairs': ['BTC/USDT', 'ETH/USDT'],
            'days': 30,
            'timeframe': '5m'
        },
        'aggressive': {
            'description': 'Aggressive - volatility coins, 14 дней',
            'pairs': ['SOL/USDT', 'AVAX/USDT', 'NEAR/USDT'],
            'days': 14,
            'timeframe': '5m'
        },
        'stable': {
            'description': 'Stable - крупные монеты, 30 дней',
            'pairs': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
            'days': 30,
            'timeframe': '15m'
        },
        'all': {
            'description': 'All available pairs - полный тест',
            'pairs': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT',
                     'AVAX/USDT', 'LINK/USDT', 'UNI/USDT'],
            'days': 30,
            'timeframe': '5m'
        }
    }

    @classmethod
    def list_profiles(cls):
        """Показать все доступные профили."""
        print("\n" + "="*70)
        print("📋 AVAILABLE BACKTEST PROFILES")
        print("="*70)

        for name, config in cls.PROFILES.items():
            print(f"\n{name}:")
            print(f"  Description: {config['description']}")
            print(f"  Pairs:       {', '.join(config['pairs'])}")
            print(f"  Days:        {config['days']}")
            print(f"  Timeframe:   {config['timeframe']}")

        print("\n" + "="*70)
        print("Usage: python scripts/run_backtest.py --profile <name>")
        print("="*70 + "\n")

    @classmethod
    def get_profile(cls, name: str):
        """Получить профиль по имени."""
        if name not in cls.PROFILES:
            print(f"❌ Unknown profile: {name}")
            print(f"Available profiles: {', '.join(cls.PROFILES.keys())}")
            cls.list_profiles()
            return None
        return cls.PROFILES[name]


class BacktestRunner:
    """Runner для запуска бэктестов."""

    def __init__(self, docker: bool = True):
        self.docker = docker
        self.container_name = "stoic_freqtrade"

    def check_data_available(self, pairs: list, timeframe: str) -> bool:
        """Проверить что данные скачаны."""
        data_dir = Path("user_data/data/binance")

        if not data_dir.exists():
            print(f"❌ Data directory not found: {data_dir}")
            return False

        missing = []
        for pair in pairs:
            pair_filename = pair.replace('/', '_')
            filename = f"{pair_filename}-{timeframe}.json"
            filepath = data_dir / filename

            if not filepath.exists():
                missing.append(pair)

        if missing:
            print(f"\n⚠️  Missing data for: {', '.join(missing)}")
            print(f"\nDownload data first:")
            print(f"  docker exec {self.container_name} freqtrade download-data \\")
            print(f"    --exchange binance \\")
            print(f"    --timeframe {timeframe} \\")
            print(f"    --pairs {' '.join(missing)} \\")
            print(f"    --days 30")
            return False

        return True

    def calculate_timerange(self, days: int) -> str:
        """Рассчитать timerange для бэктеста."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        return f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"

    def run_backtest(
        self,
        pairs: list,
        timeframe: str = '5m',
        days: int = 7,
        strategy: str = 'StoicEnsembleStrategyV2'
    ):
        """
        Запустить backtest.

        Args:
            pairs: Список пар для тестирования
            timeframe: Таймфрейм
            days: Количество дней истории
            strategy: Название стратегии
        """
        print("\n" + "="*70)
        print("🚀 STARTING BACKTEST")
        print("="*70)

        print(f"\n📊 Configuration:")
        print(f"   Strategy:   {strategy}")
        print(f"   Pairs:      {', '.join(pairs)}")
        print(f"   Timeframe:  {timeframe}")
        print(f"   Period:     {days} days")

        # Проверить данные
        if not self.check_data_available(pairs, timeframe):
            return False

        # Рассчитать timerange
        timerange = self.calculate_timerange(days)
        print(f"   Timerange:  {timerange}")

        # Создать команду
        if self.docker:
            cmd = [
                'docker', 'exec', self.container_name,
                'freqtrade', 'backtesting',
                '--strategy', strategy,
                '--timeframe', timeframe,
                '--timerange', timerange,
                '--export', 'trades'
            ]

            # Добавить пары
            for pair in pairs:
                cmd.extend(['--pairs', pair])

        else:
            cmd = [
                'freqtrade', 'backtesting',
                '--strategy', strategy,
                '--timeframe', timeframe,
                '--timerange', timerange,
                '--export', 'trades'
            ]

            for pair in pairs:
                cmd.extend(['--pairs', pair])

        print(f"\n🔄 Running backtest...")
        print(f"Command: {' '.join(cmd)}")
        print("\n" + "="*70 + "\n")

        # Запустить
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=False,
                text=True
            )

            if result.returncode == 0:
                print("\n" + "="*70)
                print("✅ BACKTEST COMPLETED SUCCESSFULLY")
                print("="*70)
                print("\nResults saved to: user_data/backtest_results/")
                print("\nView results:")
                print("  1. Open FreqUI: http://localhost:3000")
                print("  2. Go to 'Backtesting' tab")
                print("  3. Load latest results")
                print("\n" + "="*70 + "\n")
                return True
            else:
                print("\n" + "="*70)
                print("❌ BACKTEST FAILED")
                print("="*70 + "\n")
                return False

        except KeyboardInterrupt:
            print("\n\n⚠️  Backtest interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Quick backtest runner with profiles"
    )

    parser.add_argument(
        '--profile',
        type=str,
        help='Use predefined profile (quick, full, aggressive, stable, all)'
    )
    parser.add_argument(
        '--list-profiles',
        action='store_true',
        help='Show available profiles'
    )
    parser.add_argument(
        '--pair',
        type=str,
        nargs='+',
        help='Trading pairs (e.g., BTC/USDT ETH/USDT)'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        default='5m',
        help='Timeframe (default: 5m)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days to backtest (default: 7)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='StoicEnsembleStrategyV2',
        help='Strategy name (default: StoicEnsembleStrategyV2)'
    )
    parser.add_argument(
        '--no-docker',
        action='store_true',
        help='Run without Docker (use local freqtrade)'
    )

    args = parser.parse_args()

    # Показать профили
    if args.list_profiles:
        BacktestProfile.list_profiles()
        return

    runner = BacktestRunner(docker=not args.no_docker)

    # Использовать профиль или кастомные параметры
    if args.profile:
        profile = BacktestProfile.get_profile(args.profile)
        if not profile:
            return

        print(f"\n📋 Using profile: {args.profile}")
        print(f"   {profile['description']}")

        runner.run_backtest(
            pairs=profile['pairs'],
            timeframe=profile['timeframe'],
            days=profile['days'],
            strategy=args.strategy
        )

    elif args.pair:
        runner.run_backtest(
            pairs=args.pair,
            timeframe=args.timeframe,
            days=args.days,
            strategy=args.strategy
        )

    else:
        print("\n❌ Error: Specify --profile or --pair")
        print("\nExamples:")
        print("  python scripts/run_backtest.py --profile quick")
        print("  python scripts/run_backtest.py --pair BTC/USDT --days 14")
        print("\nFor help:")
        print("  python scripts/run_backtest.py --help")
        print("  python scripts/run_backtest.py --list-profiles\n")


if __name__ == "__main__":
    main()
