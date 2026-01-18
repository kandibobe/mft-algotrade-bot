import unittest

from src.risk.liquidation import LiquidationConfig, LiquidationGuard


class TestLiquidationGuardCoverage(unittest.TestCase):
    def setUp(self):
        self.config = LiquidationConfig(safety_buffer=0.1)
        self.guard = LiquidationGuard(self.config)

    def test_should_liquidate(self):
        self.assertTrue(self.guard.should_liquidate(90, 100))
        self.assertFalse(self.guard.should_liquidate(91, 100))

if __name__ == "__main__":
    unittest.main()
