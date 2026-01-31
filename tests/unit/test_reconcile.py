
import asyncio
import unittest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from scripts.ops.reconcile import Reconciler


class TestReconciler(unittest.TestCase):
    def setUp(self):
        self.reconciler = Reconciler(mode="test")
        self.reconciler.backend = AsyncMock()
        self.reconciler.risk_manager = MagicMock()
        self.reconciler.telegram = AsyncMock()

    @patch('scripts.ops.reconcile.log')
    def test_reconcile_no_discrepancy(self, mock_log):
        # Setup
        self.reconciler.risk_manager._positions = {
            "BTC/USDT": {"size": 1.0}
        }
        self.reconciler.backend.fetch_positions.return_value = [
            {"symbol": "BTC/USDT", "contracts": 1.0}
        ]
        self.reconciler.backend.fetch_balance.return_value = {}

        # Execute
        asyncio.run(self.reconciler.reconcile())

        # Verify
        self.reconciler.risk_manager.emergency_stop.assert_not_called()
        mock_log.info.assert_any_call("✅ Reconciliation successful. No discrepancies found.")

    @patch('scripts.ops.reconcile.log')
    def test_reconcile_missing_position(self, mock_log):
        # Setup: Bot thinks it has BTC, exchange says no
        self.reconciler.risk_manager._positions = {
            "BTC/USDT": {"size": 1.0}
        }
        self.reconciler.backend.fetch_positions.return_value = []
        self.reconciler.backend.fetch_balance.return_value = {}

        # Execute
        asyncio.run(self.reconciler.reconcile())

        # Verify
        self.reconciler.risk_manager.emergency_stop.assert_called_once()
        self.reconciler.telegram.send_message_async.assert_called()
        mock_log.critical.assert_called()

    @patch('scripts.ops.reconcile.log')
    def test_reconcile_ghost_position(self, mock_log):
        # Setup: Bot thinks nothing, exchange has BTC
        self.reconciler.risk_manager._positions = {}
        self.reconciler.backend.fetch_positions.return_value = [
            {"symbol": "BTC/USDT", "contracts": 1.0}
        ]
        self.reconciler.backend.fetch_balance.return_value = {}

        # Execute
        asyncio.run(self.reconciler.reconcile())

        # Verify
        self.reconciler.risk_manager.emergency_stop.assert_called_once()
        mock_log.critical.assert_called()

    def test_risk_manager_persistence(self):
        import os

        from src.risk.risk_manager import RiskManager
        db_path = "user_data/test_risk_state.db"
        if os.path.exists(db_path):
            try: os.remove(db_path)
            except: pass

        rm = RiskManager(db_path=db_path)
        rm._account_balance = Decimal("1000.0")
        rm.record_entry("BTC/USDT", 50000.0, 0.1, 49000.0, side="long")

        # New instance should load state
        rm2 = RiskManager(db_path=db_path)
        self.assertEqual(rm2._account_balance, Decimal("1000.0"))
        self.assertIn("BTC/USDT", rm2._positions)
        self.assertEqual(rm2._positions["BTC/USDT"]["size"], 0.1)

        # Cleanup (optional, windows might hold lock)

if __name__ == '__main__':
    unittest.main()
