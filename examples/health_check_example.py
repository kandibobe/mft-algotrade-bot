#!/usr/bin/env python3
"""
Example usage of the HealthCheck system.

This example shows how to:
1. Use the HealthCheck class directly
2. Run the FastAPI health check server
3. Integrate health checks with existing bot
"""

import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.monitoring.health_check import HealthCheck

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def example_direct_usage():
    """Example 1: Direct usage of HealthCheck class."""
    print("=" * 60)
    print("Example 1: Direct HealthCheck Usage")
    print("=" * 60)
    
    # Create health check instance
    health_check = HealthCheck()
    
    # Run all checks
    results = await health_check.run_all_checks()
    
    print(f"Overall Status: {results['status']}")
    print(f"Healthy: {results['healthy']}")
    print(f"Timestamp: {results['timestamp']}")
    
    print("\nIndividual Check Results:")
    for check_name, result in results['checks'].items():
        status = result.get('status', 'unknown')
        healthy = result.get('healthy', False)
        details = result.get('details', 'N/A')
        
        if isinstance(details, dict):
            details_str = ", ".join([f"{k}: {v}" for k, v in details.items()][:3])
            if len(details) > 3:
                details_str += "..."
        else:
            details_str = str(details)[:50] + "..." if len(str(details)) > 50 else str(details)
        
        print(f"  - {check_name}: {status} (healthy: {healthy})")
        print(f"    Details: {details_str}")
    
    return results

async def example_with_bot_integration():
    """Example 2: Integrating with a trading bot."""
    print("\n" + "=" * 60)
    print("Example 2: Bot Integration")
    print("=" * 60)
    
    # Create a mock bot with components
    class MockBot:
        def __init__(self):
            self.exchange = MockExchange()
            self.db = MockDatabase()
            self.ml_inference_service = MockMLInferenceService()
            self.circuit_breaker = MockCircuitBreaker()
    
    class MockExchange:
        async def fetch_ticker(self, symbol, timeout=5):
            return {'last': 50000.0, 'bid': 49900.0, 'ask': 50100.0}
        
        @property
        def name(self):
            return 'binance'
    
    class MockDatabase:
        async def execute(self, query):
            return MockResult()
    
    class MockResult:
        def scalar(self):
            return 1
    
    class MockMLInferenceService:
        models = {'ensemble_model': MockModel()}
        
        async def predict(self, model_name, features):
            return MockPrediction()
    
    class MockModel:
        feature_columns = ['rsi', 'macd', 'volume', 'close', 'volatility']
    
    class MockPrediction:
        prediction = 0.75
        signal = 'buy'
        confidence = 0.85
    
    class MockCircuitBreaker:
        def get_status(self):
            return {'state': 'closed', 'failures': 0, 'last_failure': None}
    
    # Create bot instance
    bot = MockBot()
    
    # Create health check with bot
    health_check = HealthCheck(bot=bot)
    
    # Run exchange check
    exchange_result = await health_check.check_exchange()
    print(f"Exchange Check: {exchange_result['status']}")
    
    # Run database check
    db_result = await health_check.check_database()
    print(f"Database Check: {db_result['status']}")
    
    # Run ML model check
    ml_result = await health_check.check_ml_model()
    print(f"ML Model Check: {ml_result['status']}")
    
    # Run circuit breaker check
    cb_result = await health_check.check_circuit_breaker()
    print(f"Circuit Breaker Check: {cb_result['status']}")
    
    print("\n✅ Bot integration example completed")

def example_fastapi_server():
    """Example 3: Running the FastAPI health check server."""
    print("\n" + "=" * 60)
    print("Example 3: FastAPI Health Check Server")
    print("=" * 60)
    
    try:
        from src.monitoring.health_check import FASTAPI_AVAILABLE, app
        
        if FASTAPI_AVAILABLE:
            print("FastAPI is available!")
            print("\nTo run the health check server:")
            print("  uvicorn src.monitoring.health_check:app --host 0.0.0.0 --port 8080")
            print("\nEndpoints:")
            print("  GET /health      - Liveness probe (always returns 200)")
            print("  GET /ready       - Readiness probe (checks all components)")
            print("  GET /health/detailed - Detailed health status")
            print("  GET /health/{component} - Health of specific component")
            
            print("\nKubernetes configuration example:")
            print("""
  livenessProbe:
    httpGet:
      path: /health
      port: 8080
    initialDelaySeconds: 30
    periodSeconds: 10
    
  readinessProbe:
    httpGet:
      path: /ready
      port: 8080
    initialDelaySeconds: 5
    periodSeconds: 5
            """)
        else:
            print("FastAPI is not installed.")
            print("Install it with: pip install fastapi uvicorn")
            print("\nAfter installation, you can run the server with:")
            print("  uvicorn src.monitoring.health_check:app --host 0.0.0.0 --port 8080")
    
    except ImportError as e:
        print(f"Error importing FastAPI: {e}")

async def main():
    """Run all examples."""
    print("Stoic Citadel Health Check System - Examples")
    print("=" * 60)
    
    # Example 1: Direct usage
    await example_direct_usage()
    
    # Example 2: Bot integration
    await example_with_bot_integration()
    
    # Example 3: FastAPI server
    example_fastapi_server()
    
    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
