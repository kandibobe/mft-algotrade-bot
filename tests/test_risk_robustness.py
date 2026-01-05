import unittest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st
from src.risk.position_sizing import PositionSizer, PositionSizingConfig

class TestPositionSizerRobustness(unittest.TestCase):
    def setUp(self):
        self.config = PositionSizingConfig()
        self.sizer = PositionSizer(self.config)

    @given(
        account_balance=st.floats(min_value=100.0, max_value=1000000.0),
        entry_price=st.floats(min_value=0.0001, max_value=100000.0),
        stop_loss_pct=st.floats(min_value=0.001, max_value=0.5)
    )
    def test_fixed_risk_bounds(self, account_balance, entry_price, stop_loss_pct):
        """Ensure position size is always positive and within reasonable limits."""
        stop_loss_price = entry_price * (1 - stop_loss_pct)
        
        result = self.sizer.calculate_position_size(
            account_balance=account_balance,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            method="fixed_risk"
        )
        
        # Position size must be positive
        self.assertGreater(result["position_size"], 0)
        
        # Position value should not exceed total account balance (leverage check is separate)
        # Note: Fixed risk can produce large positions if stop loss is very tight.
        # But our sizer has a max_position_pct clamp in calculate_position_size.
        self.assertLessEqual(result["position_value"], account_balance * self.config.max_position_pct + 1e-6)

    @given(
        win_rate=st.floats(min_value=0.0, max_value=1.0),
        avg_win=st.floats(min_value=0.001, max_value=1.0),
        avg_loss=st.floats(min_value=0.001, max_value=1.0)
    )
    def test_kelly_size_sanity(self, win_rate, avg_win, avg_loss):
        """Ensure Kelly sizing never returns negative or absurd values."""
        result = self.sizer.calculate_position_size(
            account_balance=10000.0,
            entry_price=100.0,
            stop_loss_price=95.0,
            method="kelly",
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss
        )
        
        self.assertGreaterEqual(result["position_size"], 0)
        self.assertLessEqual(result["position_value"], 10000.0 * self.config.max_position_pct + 1e-6)

if __name__ == "__main__":
    unittest.main()
