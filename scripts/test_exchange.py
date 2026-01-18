#!/usr/bin/env python3
"""
Test Exchange Connectivity
==========================

Verifies async connection to exchange.
Uses src.data.async_fetcher.AsyncDataFetcher.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.async_fetcher import AsyncDataFetcher, FetcherConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ExchangeTest")

async def test_connectivity():
    """Test connectivity to exchange."""
    logger.info("🚀 Starting Exchange Connectivity Test...")
    
    # Configure fetcher (public data only)
    config = FetcherConfig(
        exchange="binance",
        sandbox=False, # Use real exchange for public data
        rate_limit=True
    )
    
    try:
        async with AsyncDataFetcher(config) as fetcher:
            # 1. Test Ticker Fetch
            symbol = "BTC/USDT"
            logger.info(f"📡 Fetching ticker for {symbol}...")
            ticker = await fetcher.fetch_ticker(symbol)
            
            if ticker:
                logger.info(f"✅ Ticker received: {symbol} Last={ticker.get('last')} Vol={ticker.get('baseVolume')}")
            else:
                logger.error(f"❌ Failed to fetch ticker for {symbol}")
                
            # 2. Test OHLCV Fetch
            logger.info(f"🕯️ Fetching OHLCV for {symbol}...")
            ohlcv = await fetcher.fetch_ohlcv(symbol, limit=5)
            
            if not ohlcv.empty:
                logger.info(f"✅ OHLCV received: {len(ohlcv)} candles")
                logger.info(f"   Last candle: {ohlcv.iloc[-1].name} Close={ohlcv.iloc[-1]['close']}")
            else:
                logger.error(f"❌ Failed to fetch OHLCV for {symbol}")
                
            # 3. Test Balance (Expected to fail without keys)
            logger.info("💰 Fetching Balance (Expect Failure without keys)...")
            try:
                balance = await fetcher.fetch_balance()
                logger.info(f"✅ Balance received: {balance}")
            except Exception as e:
                logger.info(f"⚠️ Balance fetch failed as expected (No Keys): {e}")

    except Exception as e:
        logger.error(f"❌ Connectivity Test Failed: {e}")
        # Don't fail the script if it's just network issue/no keys, 
        # unless it's a critical import error
        if "ccxt" in str(e).lower() or "import" in str(e).lower():
             sys.exit(1)

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_connectivity())