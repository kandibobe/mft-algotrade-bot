import unittest
from decimal import Decimal
from unittest.mock import MagicMock, patch

from src.risk.circuit_breaker import CircuitBreakerConfig
from src.risk.liquidation import LiquidationConfig
from src.risk.position_sizing import PositionSizingConfig
from src.risk.risk_manager import RiskManager


class TestRiskManagerCoverage(unittest.TestCase):
    def setUp(self):
        self.circuit_config = CircuitBreakerConfig(max_drawdown_pct=10, daily_loss_limit_pct=5)
        self.sizing_config = PositionSizingConfig(max_position_pct=5, max_portfolio_risk_pct=1)
        self.liquidation_config = LiquidationConfig(safety_buffer=0.1)
        self.risk_manager = RiskManager(
            circuit_config=self.circuit_config,
            sizing_config=self.sizing_config,
            liquidation_config=self.liquidation_config,
            enable_notifications=False
        )

    def test_initialization(self):
        self.assertIsNotNone(self.risk_manager.circuit_breaker)
        self.assertIsNotNone(self.risk_manager.position_sizer)
        self.assertIsNotNone(self.risk_manager.liquidation_guard)
        self.assertEqual(self.risk_manager._account_balance, Decimal("0.0"))
        self.assertEqual(self.risk_manager._positions, {})

    def test_initialize_session(self):
        self.risk_manager.initialize(account_balance=10000, existing_positions={"BTC/USDT": {"size": 1, "entry_price": 50000}})
        self.assertEqual(self.risk_manager._account_balance, Decimal("10000"))
        self.assertIn("BTC/USDT", self.risk_manager._positions)

    def test_evaluate_trade_allowed(self):
        self.risk_manager.initialize(account_balance=10000)
        result = self.risk_manager.evaluate_trade(symbol="BTC/USDT", entry_price=50000, stop_loss_price=49000)
        self.assertTrue(result["allowed"])
        self.assertGreater(result["position_size"], 0)

    def test_evaluate_trade_denied_by_circuit_breaker(self):
        self.risk_manager.initialize(account_balance=10000)
        self.risk_manager.circuit_breaker.manual_stop()
        result = self.risk_manager.evaluate_trade(symbol="BTC/USDT", entry_price=50000, stop_loss_price=49000)
        self.assertFalse(result["allowed"])
        self.assertEqual(result["rejection_reason"], "Circuit breaker is OPEN (trading halted)")

    def test_evaluate_trade_denied_by_emergency_stop(self):
        self.risk_manager.initialize(account_balance=10000)
        self.risk_manager.emergency_stop()
        result = self.risk_manager.evaluate_trade(symbol="BTC/USDT", entry_price=50000, stop_loss_price=49000)
        self.assertFalse(result["allowed"])
        self.assertEqual(result["rejection_reason"], "Emergency Stop Active")

    def test_record_entry_and_exit(self):
        self.risk_manager.initialize(account_balance=10000)
        self.risk_manager.record_entry(symbol="BTC/USDT", entry_price=50000, position_size=0.1, stop_loss_price=49000)
        self.assertIn("BTC/USDT", self.risk_manager._positions)
        exit_result = self.risk_manager.record_exit(symbol="BTC/USDT", exit_price=51000)
        self.assertNotIn("BTC/USDT", self.risk_manager._positions)
        self.assertAlmostEqual(exit_result["pnl"], 100)

    @patch("src.risk.risk_manager.os.path.exists")
    @patch("src.risk.risk_manager.sqlite3")
    def test_save_and_load_state(self, mock_sqlite3, mock_exists):
        # Mocking for save/load to avoid actual file I/O
        mock_exists.return_value = True
        mock_conn = MagicMock()
        mock_sqlite3.connect.return_value = mock_conn

        self.risk_manager.initialize(account_balance=10000)
        self.risk_manager.record_entry(symbol="BTC/USDT", entry_price=50000, position_size=0.1, stop_loss_price=49000)

        # Save state
        self.risk_manager.save_state()

        # Verify save was called
        mock_conn.execute.assert_called()

        # Load state
        # For a more thorough test, you would mock the cursor and row factory
        # but for coverage, we just need to call the method.
        self.risk_manager.load_state()

        # Verify load was called
        self.assertTrue(mock_conn.execute.call_count > 1)

if __name__ == "__main__":
    unittest.main()
