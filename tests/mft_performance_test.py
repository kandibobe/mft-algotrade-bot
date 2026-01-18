import asyncio
import logging
import time

import numpy as np

from src.config.unified_config import load_config
from src.order_manager.smart_order_executor import SmartOrderExecutor
from src.websocket.aggregator import DataAggregator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MFTPerformanceTest")

class MFTPerformanceTester:
    def __init__(self):
        self.latencies = []
        self.config = load_config()
        self.aggregator = DataAggregator(self.config)
        self.executor = SmartOrderExecutor(self.config)

    async def simulate_orderbook_update(self):
        """Simulate high-frequency orderbook updates."""
        start_time = time.perf_counter()

        # Mock update logic
        # In real test, we would push to the aggregator's queue
        await asyncio.sleep(0.001) # Simulate network/processing delay

        end_time = time.perf_counter()
        self.latencies.append((end_time - start_time) * 1000) # ms

    async def run_load_test(self, duration_seconds=10, updates_per_second=100):
        """Run the load test for a specified duration."""
        logger.info(f"Starting MFT Load Test: {updates_per_second} updates/sec for {duration_seconds}s")

        start_test = time.time()
        count = 0
        while time.time() - start_test < duration_seconds:
            await self.simulate_orderbook_update()
            count += 1
            await asyncio.sleep(1.0 / updates_per_second)

        self.report_results()

    def report_results(self):
        """Print performance statistics."""
        if not self.latencies:
            logger.warning("No latency data collected.")
            return

        lat_array = np.array(self.latencies)
        logger.info("=== MFT Performance Report ===")
        logger.info(f"Total Updates: {len(self.latencies)}")
        logger.info(f"Avg Latency: {np.mean(lat_array):.4f} ms")
        logger.info(f"Min Latency: {np.min(lat_array):.4f} ms")
        logger.info(f"Max Latency: {np.max(lat_array):.4f} ms")
        logger.info(f"P95 Latency: {np.percentile(lat_array, 95):.4f} ms")
        logger.info(f"P99 Latency: {np.percentile(lat_array, 99):.4f} ms")
        logger.info("==============================")

async def main():
    tester = MFTPerformanceTester()
    await tester.run_load_test(duration_seconds=5, updates_per_second=200)

if __name__ == "__main__":
    asyncio.run(main())
