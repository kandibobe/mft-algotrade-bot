import unittest

from src.risk.position_sizing import PositionSizer, PositionSizingConfig


class TestPositionSizerCoverage(unittest.TestCase):
    def setUp(self):
        self.config = PositionSizingConfig(max_position_pct=5, max_portfolio_risk_pct=1)
        self.sizer = PositionSizer(self.config)

    def test_calculate_position_size(self):
        result = self.sizer.calculate_position_size(
            account_balance=10000,
            entry_price=50000,
            stop_loss_price=49000
        )
        self.assertGreater(result["position_size"], 0)
        self.assertLessEqual(result["position_value"], 10000 * 0.05)

    def test_zero_risk(self):
        with self.assertRaises(ValueError):
            self.sizer.calculate_position_size(
                account_balance=10000,
                entry_price=50000,
                stop_loss_price=50000
            )

if __name__ == "__main__":
    unittest.main()
