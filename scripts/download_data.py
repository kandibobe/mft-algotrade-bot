#!/usr/bin/env python3
"""
Download Data Helper
====================

Упрощенный скрипт для скачивания данных с правильными параметрами.

Usage:
    python scripts/download_data.py --pair BTC/USDT --days 30
    python scripts/download_data.py --preset major  # BTC, ETH, BNB
    python scripts/download_data.py --preset all    # Все популярные пары
"""

import argparse
import subprocess
import sys


PRESETS = {
    'major': {
        'description': 'Major coins (BTC, ETH, BNB)',
        'pairs': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    },
    'defi': {
        'description': 'DeFi tokens',
        'pairs': ['UNI/USDT', 'LINK/USDT', 'AAVE/USDT', 'CRV/USDT']
    },
    'layer1': {
        'description': 'Layer 1 platforms',
        'pairs': ['SOL/USDT', 'AVAX/USDT', 'NEAR/USDT', 'ADA/USDT']
    },
    'meme': {
        'description': 'Meme coins',
        'pairs': ['DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT']
    },
    'all': {
        'description': 'All popular pairs',
        'pairs': [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT',
            'AVAX/USDT', 'LINK/USDT', 'UNI/USDT', 'DOGE/USDT',
            'XRP/USDT', 'ADA/USDT', 'NEAR/USDT', 'PEPE/USDT'
        ]
    }
}


def download_data(
    pairs: list,
    timeframe: str = '5m',
    days: int = 30,
    use_docker: bool = True
):
    """
    Download data from exchange.

    Args:
        pairs: List of trading pairs
        timeframe: Timeframe (5m, 15m, 1h, etc.)
        days: Number of days to download
        use_docker: Use Docker container (default: True)
    """
    print(f"\n{'='*70}")
    print("📥 DOWNLOADING DATA")
    print(f"{'='*70}")

    print(f"\n📊 Configuration:")
    print(f"   Exchange:   binance")
    print(f"   Pairs:      {', '.join(pairs)}")
    print(f"   Timeframe:  {timeframe}")
    print(f"   Days:       {days}")
    print(f"   Mode:       {'Docker' if use_docker else 'Local'}")

    # Build command
    if use_docker:
        cmd = [
            'docker', 'exec', 'stoic_freqtrade',
            'freqtrade', 'download-data',
            '--exchange', 'binance',
            '--timeframe', timeframe,
            '--days', str(days)
        ]
    else:
        cmd = [
            'freqtrade', 'download-data',
            '--exchange', 'binance',
            '--timeframe', timeframe,
            '--days', str(days)
        ]

    # Add pairs
    for pair in pairs:
        cmd.extend(['--pairs', pair])

    print(f"\n🔄 Running: {' '.join(cmd)}")
    print(f"\n{'='*70}\n")

    # Execute
    try:
        result = subprocess.run(cmd, check=False)

        if result.returncode == 0:
            print(f"\n{'='*70}")
            print("✅ DATA DOWNLOADED SUCCESSFULLY")
            print(f"{'='*70}")

            print(f"\nData saved to: user_data/data/binance/")

            print(f"\nNext steps:")
            print(f"  1. Inspect data:  python scripts/inspect_data.py --pair {pairs[0]}")
            print(f"  2. Run backtest:  python scripts/run_backtest.py --profile quick")
            print(f"  3. View in FreqUI: http://localhost:3000")

            print(f"\n{'='*70}\n")
            return True

        else:
            print(f"\n{'='*70}")
            print("❌ DOWNLOAD FAILED")
            print(f"{'='*70}\n")
            return False

    except FileNotFoundError:
        print(f"\n❌ Error: {'Docker' if use_docker else 'freqtrade'} not found")
        if use_docker:
            print("Make sure Docker is installed and running:")
            print("  docker ps")
        else:
            print("Install freqtrade or use --docker flag")
        return False

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Download interrupted by user")
        return False


def list_presets():
    """Show available presets."""
    print(f"\n{'='*70}")
    print("📋 AVAILABLE PRESETS")
    print(f"{'='*70}")

    for name, config in PRESETS.items():
        print(f"\n{name}:")
        print(f"  Description: {config['description']}")
        print(f"  Pairs:       {', '.join(config['pairs'])}")

    print(f"\n{'='*70}")
    print("Usage: python scripts/download_data.py --preset <name>")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download historical data from exchange"
    )

    parser.add_argument(
        '--preset',
        type=str,
        choices=list(PRESETS.keys()),
        help='Use predefined pair preset'
    )
    parser.add_argument(
        '--list-presets',
        action='store_true',
        help='Show available presets'
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
        default=30,
        help='Number of days (default: 30)'
    )
    parser.add_argument(
        '--no-docker',
        action='store_true',
        help='Use local freqtrade instead of Docker'
    )

    args = parser.parse_args()

    if args.list_presets:
        list_presets()
        return

    # Determine pairs
    if args.preset:
        preset = PRESETS[args.preset]
        print(f"\n📋 Using preset: {args.preset}")
        print(f"   {preset['description']}")
        pairs = preset['pairs']

    elif args.pair:
        pairs = args.pair

    else:
        print("\n❌ Error: Specify --preset or --pair")
        print("\nExamples:")
        print("  python scripts/download_data.py --preset major")
        print("  python scripts/download_data.py --pair BTC/USDT ETH/USDT --days 30")
        print("\nFor help:")
        print("  python scripts/download_data.py --help")
        print("  python scripts/download_data.py --list-presets\n")
        return

    # Download
    download_data(
        pairs=pairs,
        timeframe=args.timeframe,
        days=args.days,
        use_docker=not args.no_docker
    )


if __name__ == "__main__":
    main()
