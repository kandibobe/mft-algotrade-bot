import unittest

import numpy as np
import pandas as pd

from src.risk.hrp import get_hrp_weights


class TestHRP(unittest.TestCase):
    def test_hrp_weights(self):
        # Create correlated price data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100)

        # Asset 1 and 2 are highly correlated
        a1 = np.random.normal(0.001, 0.02, 100).cumsum() + 100
        a2 = a1 + np.random.normal(0, 0.001, 100)

        # Asset 3 is independent
        a3 = np.random.normal(0.001, 0.02, 100).cumsum() + 100

        df = pd.DataFrame({
            'BTC/USDT': a1,
            'ETH/USDT': a2,
            'SOL/USDT': a3
        }, index=dates)

        weights = get_hrp_weights(df)

        # Check that weights sum to ~1.0
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)

        # Correlated assets (a1, a2) should together have roughly same weight as independent (a3)
        # because HRP balances risk between clusters
        correlated_weight = weights['BTC/USDT'] + weights['ETH/USDT']
        independent_weight = weights['SOL/USDT']

        print(f"\nHRP Weights: {weights}")
        self.assertTrue(0.3 < independent_weight < 0.7)

if __name__ == "__main__":
    unittest.main()
