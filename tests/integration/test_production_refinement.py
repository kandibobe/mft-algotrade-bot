"""
Production Integration Test Suite (Refined)
===========================================

Comprehensive integration tests to verify refined production modules.
"""

import asyncio
import unittest
import os
import uuid
import sqlite3
from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, AsyncMock

# Set ENV to something other than production to allow mock use
os.environ["ENV"] = "testing"

from src.risk.risk_manager import RiskManager
from src.order_manager.order_ledger import OrderLedger
from src.ml.feature_store import RedisFeatureStore

class TestProductionRefinement(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        # Use temporary DBs for testing
        test_id = uuid.uuid4().hex[:8]
        self.risk_db = f"user_data/test_risk_{test_id}.db"
        self.order_db = f"data/test_orders_{test_id}.db"
        
        self.risk_manager = RiskManager(db_path=self.risk_db)
        self.ledger = OrderLedger(db_path=self.order_db)
        # RedisFeatureStore will use mock mode
        self.fs = RedisFeatureStore(use_mock=True)

    async def asyncTearDown(self):
        # Clean up test databases
        self.risk_manager = None
        self.ledger = None
        
        await asyncio.sleep(0.2)
        
        def try_remove(path):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

        try_remove(self.risk_db)
        try_remove(self.order_db)

    def test_risk_manager_atomic_updates(self):
        """Verify RiskManager correctly calculates equity and exposure."""
        self.risk_manager.update_balance("binance", 10000.0)
        self.risk_manager.record_entry("binance", "BTC/USDT", 0.1, 50000.0, "long")
        
        status = self.risk_manager.get_status()
        self.assertEqual(status['balances']['binance'], 10000.0)
        # In refined RiskManager, open_positions is across all exchanges
        self.assertEqual(status['metrics']['open_positions'], 1)
        # 0.1 BTC * 50000.0 = 5000.0 exposure
        self.assertEqual(float(status['metrics']['total_exposure']), 5000.0)
        
        # Test uPnL update
        self.risk_manager.update_position_pnl("binance", "BTC/USDT", 51000.0, 100.0)
        status = self.risk_manager.get_status()
        self.assertEqual(float(status['metrics']['unrealized_pnl']), 100.0)
        self.assertEqual(float(status['metrics']['equity']), 10100.0)

    async def test_risk_safety_gate(self):
        """Test the evaluate_safety atomic check."""
        self.risk_manager.update_balance("binance", 1000.0)
        # Manually trip circuit breaker
        self.risk_manager.circuit_breaker.manual_stop()
        
        allowed, reason = await self.risk_manager.evaluate_safety("BTC/USDT", "long", 0.1, 50000.0)
        self.assertFalse(allowed)
        self.assertIn("Circuit Breaker", reason)

    async def test_order_ledger_reconciliation(self):
        """Test OrderLedger correctly handles exchange reconciliation."""
        # 1. Record an order locally
        order_id = f"ord_{uuid.uuid4().hex[:6]}"
        order = {
            "order_id": order_id,
            "symbol": "BTC/USDT", 
            "side": "buy", 
            "quantity": 0.1, 
            "exchange_order_id": "exch_123", 
            "status": "pending"
        }
        self.ledger.store_order(order, f"idemp_{order_id}")
        
        # 2. Mock exchange client
        exchange_mock = AsyncMock()
        exchange_mock.fetch_order.return_value = {
            "id": "exch_123",
            "status": "filled",
            "filled": 0.1
        }
        
        # 3. Reconcile
        await self.ledger.reconcile_with_exchange(exchange_mock)
        
        # 4. Verify - OrderLedger refined uses 'filled' instead of 'closed' status for success
        updated = self.ledger.get_order_by_exchange_id("exch_123")
        self.assertIsNotNone(updated, "Order should be found by exchange ID")
        self.assertEqual(updated['status'], "filled")
        self.assertEqual(updated['filled_quantity'], 0.1)

    async def test_feature_store_mock_behavior(self):
        """Test RedisFeatureStore mock behavior."""
        # Mocking redis get to return None to force mock generation
        self.fs._redis.get = MagicMock(return_value=None)
        features = await self.fs.get_online_features("BTC/USDT", ["rsi", "ema"])
        self.assertEqual(len(features), 1)
        self.assertIn("rsi", features.columns)

if __name__ == "__main__":
    unittest.main()