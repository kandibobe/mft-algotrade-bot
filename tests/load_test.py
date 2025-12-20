"""
Load testing for trading bot API endpoints using Locust.

This module defines a Locust user class that simulates trading bot API traffic
to identify performance bottlenecks and test system scalability.

Usage:
    locust -f tests/load_test.py --host http://localhost:8080
    # Open browser at http://localhost:8089 to start test

Example with specific parameters:
    locust -f tests/load_test.py --host http://localhost:8080 --users 100 --spawn-rate 10 --run-time 1h

Dependencies:
    pip install locust
"""
import locust
from locust import HttpUser, task, between


class TradingBotUser(HttpUser):
    """Locust user that simulates trading bot API requests."""
    
    # Wait between 1 and 3 seconds between tasks
    wait_time = between(1, 3)
    
    @task(3)  # Higher weight: 3x more likely to be called
    def get_signal(self):
        """Test ML inference endpoint for signal generation."""
        self.client.post("/api/v1/signal", json={
            "symbol": "BTC/USDT",
            "timeframe": "5m"
        })
        
    @task(1)  # Lower weight: 1x
    def place_order(self):
        """Test order placement endpoint."""
        self.client.post("/api/v1/orders", json={
            "symbol": "BTC/USDT",
            "side": "buy",
            "quantity": 0.001,
            "order_type": "market"
        })
    
    @task(2)
    def health_check(self):
        """Test health check endpoint (optional)."""
        self.client.get("/health")
    
    def on_start(self):
        """Called when a user starts."""
        self.client.headers = {
            "Content-Type": "application/json",
            "User-Agent": "Locust-TradingBot-Test/1.0"
        }


if __name__ == "__main__":
    # For local testing without locust command line
    import os
    print("This file is meant to be run with locust command:")
    print("  locust -f tests/load_test.py --host http://localhost:8080")
    print("\nTo install locust: pip install locust")
    print("\nTypical load test scenario:")
    print("  Simulate 100 users with 10 requests/sec to find bottlenecks")
