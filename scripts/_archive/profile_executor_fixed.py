
import cProfile
import pstats
import asyncio
from src.order_manager.smart_order_executor import SmartOrderExecutor
from src.order_manager.smart_order import ChaseLimitOrder
from src.websocket.aggregator import AggregatedTicker
from unittest.mock import MagicMock, AsyncMock

async def main():
    # Mock aggregator
    aggregator = MagicMock()
    aggregator.get_aggregated_ticker.return_value = AggregatedTicker(
        symbol="BTC/USDT",
        best_bid=49999.0,
        best_ask=50000.0,
        best_bid_exchange="mock",
        best_ask_exchange="mock",
        spread=1.0,
        spread_pct=0.002,
        exchanges={},
        vwap=50000,
        total_volume_24h=1000,
        timestamp=0
    )

    # Mock Risk Manager to always allow
    risk_manager = MagicMock()
    risk_manager.circuit_breaker.can_trade.return_value = True
    risk_manager.evaluate_trade.return_value = {"allowed": True, "position_size": 0.01, "position_value": 500}

    executor = SmartOrderExecutor(aggregator=aggregator, dry_run=True, risk_manager=risk_manager)
    
    order = ChaseLimitOrder(
        symbol="BTC/USDT",
        side="buy",
        quantity=0.01,
        price=50000.0,
        chase_offset=10,
        max_chase_price=50010.0,
        stop_price=49000.0
    )

    # Performance measurement
    import time
    start_time = time.time()
    
    profiler = cProfile.Profile()
    profiler.enable()

    await executor.submit_order(order)

    profiler.disable()
    
    end_time = time.time()
    print(f"\nTotal submission time: {(end_time - start_time)*1000:.2f} ms")
    
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20)

if __name__ == "__main__":
    asyncio.run(main())