import unittest

from src.risk.circuit_breaker import CircuitBreaker, CircuitBreakerConfig


class TestCircuitBreakerCoverage(unittest.TestCase):
    def setUp(self):
        self.config = CircuitBreakerConfig(max_drawdown_pct=10, daily_loss_limit_pct=5)
        self.circuit_breaker = CircuitBreaker(self.config)

    def test_initial_state(self):
        self.assertTrue(self.circuit_breaker.can_trade())
        status = self.circuit_breaker.get_status()
        self.assertEqual(status["state"], "CLOSED")

    def test_manual_stop(self):
        self.circuit_breaker.manual_stop()
        self.assertFalse(self.circuit_breaker.can_trade())
        status = self.circuit_breaker.get_status()
        self.assertEqual(status["state"], "MANUAL_OPEN")

    def test_daily_loss_limit(self):
        self.circuit_breaker.initialize_session(1000)
        self.circuit_breaker.record_trade({}, -0.06) # 6% loss
        self.assertFalse(self.circuit_breaker.can_trade())
        status = self.circuit_breaker.get_status()
        self.assertEqual(status["state"], "DAILY_LOSS_LIMIT_OPEN")

    def test_drawdown_limit(self):
        self.circuit_breaker.initialize_session(1000)
        self.circuit_breaker.session.current_balance = 890 # 11% drawdown
        self.circuit_breaker.record_trade({}, -0.01) # 1% loss
        self.assertFalse(self.circuit_breaker.can_trade())
        status = self.circuit_breaker.get_status()
        self.assertEqual(status["state"], "MAX_DRAWDOWN_OPEN")

if __name__ == "__main__":
    unittest.main()
