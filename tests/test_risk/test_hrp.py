"""
Tests for Hierarchical Risk Parity (HRP) Optimization.
"""

import unittest

import numpy as np
import pandas as pd

from src.risk import hrp


class TestHRP(unittest.TestCase):
    def setUp(self):
        # Create synthetic price data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.prices = pd.DataFrame({
            'BTC/USDT': np.random.randn(100).cumsum() + 100,
            'ETH/USDT': np.random.randn(100).cumsum() + 50,
            'SOL/USDT': np.random.randn(100).cumsum() + 20
        }, index=dates)

        # Ensure no negative prices
        self.prices = self.prices.abs() + 1.0

    def test_get_hrp_weights_normal(self):
        """Test HRP weights calculation with normal data."""
        weights = hrp.get_hrp_weights(self.prices)

        # Check output structure
        self.assertIsInstance(weights, dict)
        self.assertEqual(len(weights), 3)
        self.assertIn('BTC/USDT', weights)

        # Check weights sum to 1.0 (approx)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=4)

        # Check weights are positive
        for w in weights.values():
            self.assertGreater(w, 0.0)

    def test_get_hrp_weights_insufficient_data(self):
        """Test fallback when not enough data points."""
        short_prices = self.prices.iloc[:5]
        weights = hrp.get_hrp_weights(short_prices)

        # Should return equal weights
        self.assertAlmostEqual(weights['BTC/USDT'], 1.0/3, places=4)

    def test_get_hrp_weights_single_asset(self):
        """Test handling of single asset."""
        single_price = self.prices[['BTC/USDT']]
        weights = hrp.get_hrp_weights(single_price)

        # Should return weight 1.0 for single asset (or handle gracefully)
        # Based on implementation, get_hrp_weights checks for len(prices.columns) < 2
        self.assertEqual(len(weights), 1)
        self.assertEqual(weights['BTC/USDT'], 1.0)

    def test_get_hrp_weights_with_nan(self):
        """Test robustness against NaN values."""
        # Introduce NaNs
        nan_prices = self.prices.copy()
        nan_prices.iloc[10:15, 0] = np.nan

        # Method should handle it (by dropna inside usually)
        # Checking implementation: src/risk/hrp.py uses returns.dropna()
        weights = hrp.get_hrp_weights(nan_prices)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=4)

if __name__ == '__main__':
    unittest.main()
