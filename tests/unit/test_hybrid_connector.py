"""
Unit Tests for Hybrid Connector
===============================

Tests the `HybridConnectorMixin` logic, including initialization,
threading, and data retrieval.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch, ANY
from src.strategies.hybrid_connector import HybridConnectorMixin
from src.websocket.aggregator import AggregatedTicker

# Mocking external dependencies
@pytest.fixture
def mock_strategy():
    """Create a mock strategy class that inherits from HybridConnectorMixin."""
    class MockStrategy(HybridConnectorMixin):
        def __init__(self):
            self.dp = MagicMock()
            self.dp.runmode.value = "live"
    return MockStrategy()

@pytest.fixture
def mock_config_manager():
    with patch("src.config.manager.ConfigurationManager") as mock:
        mock.get_config.return_value.exchange.name = "binance"
        mock.get_config.return_value.exchange.api_key = "key"
        mock.get_config.return_value.exchange.api_secret = "secret"
        mock.get_config.return_value.dry_run = True
        yield mock

@pytest.mark.asyncio
async def test_initialization(mock_strategy, mock_config_manager):
    """Test that the connector initializes correctly."""
    with patch("src.strategies.hybrid_connector.DataAggregator") as MockAgg, \
         patch("src.strategies.hybrid_connector.SmartOrderExecutor") as MockExec, \
         patch("threading.Thread") as MockThread:
        
        mock_strategy.initialize_hybrid_connector(["BTC/USDT"])
        
        assert mock_strategy._loop is not None
        assert mock_strategy._aggregator is not None
        assert mock_strategy._executor is not None
        
        # Verify Aggregator setup
        mock_strategy._aggregator.add_exchange.assert_called()
        
        # Verify Thread start
        MockThread.assert_called_once()
        mock_strategy._thread.start.assert_called_once()

def test_backtest_mode_skip(mock_strategy):
    """Test that connector skips initialization in backtest mode."""
    mock_strategy.dp.runmode.value = "backtest"
    
    with patch("src.strategies.hybrid_connector.DataAggregator") as MockAgg:
        mock_strategy.initialize_hybrid_connector(["BTC/USDT"])
        MockAgg.assert_not_called()
        assert mock_strategy._aggregator is None

def test_get_realtime_metrics(mock_strategy):
    """Test data retrieval from cache."""
    # Manually inject data into cache
    ticker = AggregatedTicker(
        symbol="BTC/USDT",
        best_bid=50000.0,
        best_bid_exchange="binance",
        best_ask=50010.0,
        best_ask_exchange="binance",
        spread=10.0,
        spread_pct=0.0002,
        exchanges={},
        vwap=50005.0,
        total_volume_24h=100.0,
        timestamp=1234567890.0,
        imbalance=0.1
    )
    mock_strategy._metrics_cache["BTC/USDT"] = ticker
    
    # Test retrieval
    result = mock_strategy.get_realtime_metrics("BTC_USDT")
    assert result == ticker
    assert result.best_bid == 50000.0

def test_market_safety_check(mock_strategy):
    """Test the safety check logic."""
    # Case 1: No data -> Safe (default to allow, or warn)
    assert mock_strategy.check_market_safety("ETH/USDT", "long") is True
    
    # Common dummy args
    dummy_args = {
        "best_bid": 100.0, "best_bid_exchange": "e1",
        "best_ask": 101.0, "best_ask_exchange": "e1",
        "spread": 1.0, "spread_pct": 0.01,
        "exchanges": {}, "vwap": 100.5, "total_volume_24h": 1000,
        "timestamp": 1234567890.0
    }

    # Case 2: Unreliable data
    bad_ticker = AggregatedTicker(symbol="ETH/USDT", is_reliable=False, **dummy_args)
    mock_strategy._metrics_cache["ETH/USDT"] = bad_ticker
    assert mock_strategy.check_market_safety("ETH_USDT", "long") is False
    
    # Case 3: High spread
    wide_ticker = AggregatedTicker(symbol="ETH/USDT", is_reliable=True, **{**dummy_args, "spread_pct": 0.6}) # > 0.5
    mock_strategy._metrics_cache["ETH/USDT"] = wide_ticker
    assert mock_strategy.check_market_safety("ETH_USDT", "long") is False
    
    # Case 4: Good data
    good_ticker = AggregatedTicker(symbol="ETH/USDT", is_reliable=True, **{**dummy_args, "spread_pct": 0.01})
    mock_strategy._metrics_cache["ETH/USDT"] = good_ticker
    assert mock_strategy.check_market_safety("ETH_USDT", "long") is True
