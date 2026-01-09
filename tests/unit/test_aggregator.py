import pytest
import asyncio
from unittest.mock import MagicMock
from src.websocket.aggregator import DataAggregator
from src.websocket.data_types import OrderbookData, TickerData

@pytest.mark.asyncio
async def test_aggregator_imbalance_calculation():
    aggregator = DataAggregator()
    
    # Need a ticker to make the symbol active in aggregator
    ticker_data = TickerData(
        exchange="binance",
        symbol="BTC/USDT",
        bid=50000.0, ask=50100.0, last=50050.0,
        volume_24h=1000.0, change_24h=0.0, timestamp=123456789.0
    )
    await aggregator._process_ticker(ticker_data)

    # Mock orderbook data with high bid volume (positive imbalance)
    ob_data = OrderbookData(
        exchange="binance",
        symbol="BTC/USDT",
        bids=[[50000.0, 10.0], [49990.0, 5.0]],
        asks=[[50100.0, 1.0], [50200.0, 1.0]],
        timestamp=123456789.0,
        imbalance=0.76  # (15 - 2) / (15 + 2) = 13/17 ≈ 0.76
    )
    
    await aggregator._process_orderbook(ob_data)
    
    ticker = aggregator.get_aggregated_ticker("BTC/USDT")
    assert ticker is not None
    assert ticker.imbalance == 0.76

@pytest.mark.asyncio
async def test_aggregator_multi_exchange_imbalance():
    aggregator = DataAggregator()

    # Tickers for both
    await aggregator._process_ticker(TickerData("binance", "BTC/USDT", 50000, 50100, 50050, 100, 0, 0))
    await aggregator._process_ticker(TickerData("bybit", "BTC/USDT", 50001, 50101, 50051, 100, 0, 0))
    
    ob_binance = OrderbookData(
        exchange="binance",
        symbol="BTC/USDT",
        bids=[], asks=[], timestamp=0,
        imbalance=0.5
    )
    ob_bybit = OrderbookData(
        exchange="bybit",
        symbol="BTC/USDT",
        bids=[], asks=[], timestamp=0,
        imbalance=-0.5
    )
    
    await aggregator._process_orderbook(ob_binance)
    await aggregator._process_orderbook(ob_bybit)
    
    ticker = aggregator.get_aggregated_ticker("BTC/USDT")
    assert ticker.imbalance == 0.0 # Average of 0.5 and -0.5
