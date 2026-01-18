import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from src.order_manager.order_types import OrderStatus
from src.order_manager.smart_order import ChaseLimitOrder
from src.order_manager.smart_order_executor import SmartOrderExecutor
from src.strategies.hybrid_connector import HybridConnectorMixin
from src.websocket.aggregator import AggregatedTicker


class MockStrategy(HybridConnectorMixin):
    def __init__(self):
        self.dp = MagicMock()
        self.dp.runmode.value = "live"

@pytest.fixture
def strategy():
    return MockStrategy()

@pytest.mark.asyncio
async def test_hybrid_flow_ticker_to_cache(strategy):
    """Test that tickers from aggregator reach the strategy cache"""
    with patch('src.strategies.hybrid_connector.DataAggregator') as MockAggregator, \
         patch('src.strategies.hybrid_connector.SmartOrderExecutor') as MockExecutor:

        # Setup mocks
        aggregator_instance = MockAggregator.return_value
        strategy.initialize_hybrid_connector(['BTC/USDT'], 'binance')

        # Simulate a ticker update
        ticker = AggregatedTicker(
            symbol='BTC/USDT',
            best_bid=50000.0,
            best_bid_exchange='binance',
            best_ask=50010.0,
            best_ask_exchange='binance',
            spread=10.0,
            spread_pct=0.02,
            exchanges={},
            vwap=50005.0,
            total_volume_24h=1000.0,
            timestamp=time.time()
        )

        # Trigger the callback manually (since it's registered during initialize)
        callback = aggregator_instance.on_aggregated_ticker.call_args[0][0]
        await callback(ticker)

        # Verify cache
        cached = strategy.get_realtime_metrics('BTC/USDT')
        assert cached is not None
        assert cached.vwap == 50005.0
        assert cached.spread_pct == 0.02

@pytest.mark.asyncio
async def test_check_market_safety_spread(strategy):
    """Test that safety check correctly identifies high spread"""
    ticker = AggregatedTicker(
        symbol='BTC/USDT',
        best_bid=50000.0,
        best_bid_exchange='binance',
        best_ask=51000.0,
        best_ask_exchange='binance',
        spread=1000.0,
        spread_pct=2.0, # 2% spread
        exchanges={},
        vwap=50500.0,
        total_volume_24h=1000.0,
        timestamp=time.time()
    )

    with strategy._cache_lock:
        strategy._metrics_cache['BTC/USDT'] = ticker

    # High spread should return False
    assert strategy.check_market_safety('BTC/USDT', 'buy') is False

@pytest.mark.asyncio
async def test_check_market_safety_normal(strategy):
    """Test that safety check allows trading with normal spread"""
    ticker = AggregatedTicker(
        symbol='BTC/USDT',
        best_bid=50000.0,
        best_bid_exchange='binance',
        best_ask=50010.0,
        best_ask_exchange='binance',
        spread=10.0,
        spread_pct=0.02,
        exchanges={},
        vwap=50005.0,
        total_volume_24h=1000.0,
        timestamp=time.time()
    )

    with strategy._cache_lock:
        strategy._metrics_cache['BTC/USDT'] = ticker

    assert strategy.check_market_safety('BTC/USDT', 'buy') is True

@pytest.mark.asyncio
async def test_slippage_check_rejection(strategy):
    """Test that executor rejects order with high slippage"""
    mock_aggregator = MagicMock()
    ticker = AggregatedTicker(
        symbol='BTC/USDT',
        best_bid=50000.0,
        best_bid_exchange='binance',
        best_ask=50000.0, # Best ask is 50000
        best_ask_exchange='binance',
        spread=0.0,
        spread_pct=0.0,
        exchanges={},
        vwap=50000.0,
        total_volume_24h=1000.0,
        timestamp=time.time()
    )
    mock_aggregator.get_aggregated_ticker.return_value = ticker

    executor = SmartOrderExecutor(aggregator=mock_aggregator, dry_run=True)

    order = ChaseLimitOrder(
        symbol='BTC/USDT',
        side='buy',
        quantity=1.0,
        price=51000.0 # 2% slippage from 50000
    )

    with pytest.raises(RuntimeError, match="High slippage detected"):
        await executor.submit_order(order)

@pytest.mark.asyncio
async def test_shadow_mode_execution(strategy):
    """Test that shadow mode simulates fill and logs to DB"""
    mock_aggregator = MagicMock()
    ticker = AggregatedTicker(
        symbol='BTC/USDT',
        best_bid=50000.0,
        best_bid_exchange='binance',
        best_ask=50000.0,
        best_ask_exchange='binance',
        spread=0.0,
        spread_pct=0.0,
        exchanges={},
        vwap=50000.0,
        total_volume_24h=1000.0,
        timestamp=time.time()
    )
    mock_aggregator.get_aggregated_ticker.return_value = ticker

    # We need to mock DatabaseManager to avoid real DB connection in test
    with patch('src.database.db_manager.DatabaseManager') as MockDB:
        executor = SmartOrderExecutor(aggregator=mock_aggregator, shadow_mode=True)
        executor.risk_manager.circuit_breaker.manual_reset()
        executor._running = True # Simulate started executor

        order = ChaseLimitOrder(
            symbol='BTC/USDT',
            side='buy',
            quantity=1.0,
            price=50000.0
        )
        order.signal_timestamp = time.time()
        order.attribution_metadata = {"strategy_name": "test_strat"}

        # In shadow mode, submit_order starts _manage_order which calls _execute_standard_order
        await executor.submit_order(order)

        # Force one iteration of _execute_standard_order logic if needed,
        # but since we mocked ticker to match, it should fill immediately in the task.

        # Wait a bit for the async task to process
        await asyncio.sleep(0.1)

        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 1.0
    assert order.average_fill_price == 50000.0

    # Verify DB logging was attempted
    assert MockDB.return_value.session.called

@pytest.mark.asyncio
async def test_latency_tracking(strategy):
    """Test that signal, submission, and fill timestamps are tracked"""
    mock_aggregator = MagicMock()
    ticker = AggregatedTicker(
        symbol='BTC/USDT',
        best_bid=50000.0,
        best_bid_exchange='binance',
        best_ask=50000.0,
        best_ask_exchange='binance',
        spread=0.0,
        spread_pct=0.0,
        exchanges={},
        vwap=50000.0,
        total_volume_24h=1000.0,
        timestamp=time.time()
    )
    mock_aggregator.get_aggregated_ticker.return_value = ticker

    with patch('src.database.db_manager.DatabaseManager'):
        executor = SmartOrderExecutor(aggregator=mock_aggregator, shadow_mode=True)
        executor.risk_manager.circuit_breaker.manual_reset()
        executor._running = True

        order = ChaseLimitOrder(
            symbol='BTC/USDT',
            side='buy',
            quantity=1.0,
            price=50000.0
        )
        order.attribution_metadata = {"strategy_name": "test_strat"}
        # 1. Signal generated
        order.signal_timestamp = time.time()

        # 2. Submit order (sets submission_timestamp)
        await executor.submit_order(order)

        # Wait for fill (sets fill_timestamp in shadow mode matching loop)
        await asyncio.sleep(0.1)

        assert order.signal_timestamp is not None
        assert order.submission_timestamp is not None
        assert order.fill_timestamp is not None
    assert order.submission_timestamp >= order.signal_timestamp
    assert order.fill_timestamp >= order.submission_timestamp

@pytest.mark.asyncio
async def test_multi_exchange_risk_aggregation(strategy):
    """Test that risk manager aggregates exposure across multiple exchanges"""
    from src.risk.risk_manager import RiskManager

    rm = RiskManager()
    rm.circuit_breaker.manual_reset()

    # Initialize two exchanges
    rm.initialize(account_balance=10000.0, exchange="binance")
    rm.initialize(account_balance=5000.0, exchange="bybit")

    # Total balance should be 15000
    assert float(rm._account_balance) == 15000.0

    # Record entries on different exchanges
    rm.record_entry("BTC/USDT", entry_price=50000.0, position_size=0.1, stop_loss_price=45000.0, exchange="binance")
    rm.record_entry("ETH/USDT", entry_price=3000.0, position_size=1.0, stop_loss_price=2800.0, exchange="bybit")

    metrics = rm.get_metrics()
    # BTC value = 5000, ETH value = 3000. Total = 8000
    assert metrics['total_exposure'] == 8000.0
    assert metrics['open_positions'] == 2

    # Evaluate a new trade. We set max_risk_pct high to ensure it returns sizing_details
    # instead of just rejection if portfolio exposure is hit.

    rm.position_sizer.config.max_portfolio_risk = 0.9 # 90%
    rm.position_sizer.config.max_position_pct = 0.9 # 90%
    rm.position_sizer.config.max_exposure_pct = 0.9 # 90%

    res = rm.evaluate_trade("SOL/USDT", entry_price=100.0, stop_loss_price=95.0, exchange="binance", force_allowed=True)

    if not res["allowed"]:
        pytest.fail(f"Trade should have been allowed with force_allowed=True, but was rejected: {res['rejection_reason']}")

    # Check that it uses aggregated balance
    assert float(rm._account_balance) == 15000.0

@pytest.mark.asyncio
async def test_hard_stop_loss_trigger(strategy):
    """Test that Hard Stop Loss Service triggers close order on high drawdown"""
    from src.risk.hard_stop_service import HardStopLossService
    mock_backend = MagicMock()

    # Setup AsyncMocks
    mock_backend.fetch_positions = asyncio.iscoroutinefunction(MagicMock())
    async def async_fetch():
        return [
            {
                "symbol": "BTC/USDT",
                "entry_price": 50000.0,
                "current_price": 40000.0, # 20% loss
                "side": "long",
                "amount": 1.0
            }
        ]
    mock_backend.fetch_positions = async_fetch

    order_called = asyncio.Event()
    captured_args = []
    async def async_order(*args, **kwargs):
        captured_args.append((args, kwargs))
        order_called.set()
        return {"id": "emergency_id"}

    mock_backend.create_limit_sell_order = async_order

    service = HardStopLossService(backend=mock_backend, max_drawdown_per_trade=0.10)
    await service._check_positions()

    assert order_called.is_set()
    args, kwargs = captured_args[0]
    assert args[0] == "BTC/USDT"
    assert kwargs["params"]["reduceOnly"] is True

@pytest.mark.asyncio
async def test_data_quality_outlier_rejection(strategy):
    """Test that aggregator rejects extreme price outliers"""
    from src.websocket.aggregator import DataAggregator
    from src.websocket.data_stream import TickerData

    aggregator = DataAggregator()
    symbol = "BTC/USDT"
    exchange = "binance"

    # 1. First valid tick
    tick1 = TickerData(symbol=symbol, exchange=exchange, last=50000.0, bid=49990.0, ask=50010.0, volume_24h=100.0, change_24h=0.0, timestamp=time.time())
    await aggregator._process_ticker(tick1)
    assert symbol in aggregator._tickers
    assert aggregator._tickers[symbol][exchange].last == 50000.0

    # 2. Outlier tick (50% jump)
    tick2 = TickerData(symbol=symbol, exchange=exchange, last=75000.0, bid=74990.0, ask=75010.0, volume_24h=100.0, change_24h=0.0, timestamp=time.time())
    await aggregator._process_ticker(tick2)

    # Price should still be 50000 in storage
    assert aggregator._tickers[symbol][exchange].last == 50000.0

@pytest.mark.asyncio
async def test_stale_data_reliability(strategy):
    """Test that aggregated ticker marks stale data as unreliable"""
    from src.websocket.aggregator import DataAggregator
    from src.websocket.data_stream import TickerData

    aggregator = DataAggregator()
    symbol = "BTC/USDT"
    exchange = "binance"

    # Add a tick from 10 seconds ago
    stale_time = time.time() - 10.0
    tick = TickerData(symbol=symbol, exchange=exchange, last=50000.0, bid=49990.0, ask=50010.0, volume_24h=100.0, change_24h=0.0, timestamp=stale_time)
    await aggregator._process_ticker(tick)

    aggregated = aggregator.get_aggregated_ticker(symbol)
    assert aggregated.is_reliable is False
    assert "Stale data" in aggregated.reliability_reason
