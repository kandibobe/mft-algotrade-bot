#!/usr/bin/env python3
"""
Simple test for HealthCheck functionality without external dependencies.
"""

import asyncio
import sys
from unittest.mock import Mock, AsyncMock, patch

# Mock the imports before importing HealthCheck
sys.modules['fastapi'] = Mock()
sys.modules['fastapi'].FastAPI = Mock
sys.modules['fastapi'].HTTPException = Exception
sys.modules['fastapi.responses'] = Mock()
sys.modules['fastapi.responses'].JSONResponse = Mock

# Mock other dependencies
sys.modules['src.monitoring.metrics_exporter'] = Mock()
sys.modules['src.monitoring.metrics_exporter'].get_exporter = Mock()
sys.modules['config.database'] = Mock()
sys.modules['config.database'].get_session = Mock()
sys.modules['src.risk.circuit_breaker'] = Mock()
sys.modules['src.risk.circuit_breaker'].CircuitBreaker = Mock()
sys.modules['src.ml.inference_service'] = Mock()
sys.modules['src.ml.inference_service'].MLInferenceService = Mock()

# Now import HealthCheck
from src.monitoring.health_check import HealthCheck

async def test_health_check_with_mocks():
    """Test HealthCheck with mocked dependencies."""
    print("Testing HealthCheck with mocked dependencies...")
    
    # Create a mock bot
    mock_bot = Mock()
    mock_bot.exchange = Mock()
    mock_bot.exchange.fetch_ticker = AsyncMock(return_value={'last': 50000.0})
    mock_bot.exchange.name = 'binance'
    
    # Create health check with mock bot
    health_check = HealthCheck(bot=mock_bot)
    
    # Mock the clients that were initialized
    health_check.exchange_client = mock_bot.exchange
    
    # Mock database client with proper async context manager
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_session.execute.return_value.scalar = AsyncMock(return_value=1)
    
    mock_db_context = AsyncMock()
    mock_db_context.__aenter__ = AsyncMock(return_value=mock_session)
    mock_db_context.__aexit__ = AsyncMock(return_value=None)
    
    health_check.db_client = Mock(return_value=mock_db_context)
    
    health_check.ml_client = Mock()
    health_check.ml_client.models = {'test_model': Mock(feature_columns=['rsi', 'macd', 'volume'])}
    health_check.ml_client.predict = AsyncMock(return_value=Mock(prediction=1.0, signal='buy', confidence=0.8))
    
    health_check.circuit_breaker = Mock()
    health_check.circuit_breaker.get_status = Mock(return_value={'state': 'closed', 'failures': 0})
    
    health_check.redis_client = None  # Not available
    
    print("\n1. Testing exchange connection check with mock...")
    exchange_result = await health_check.check_exchange()
    print(f"   Result: {exchange_result['status']}")
    print(f"   Healthy: {exchange_result['healthy']}")
    assert exchange_result['status'] == 'healthy'
    
    print("\n2. Testing database connection check with mock...")
    db_result = await health_check.check_database()
    print(f"   Result: {db_result['status']}")
    print(f"   Healthy: {db_result['healthy']}")
    assert db_result['status'] == 'healthy'
    
    print("\n3. Testing ML model check with mock...")
    ml_result = await health_check.check_ml_model()
    print(f"   Result: {ml_result['status']}")
    print(f"   Healthy: {ml_result['healthy']}")
    assert ml_result['status'] == 'healthy'
    
    print("\n4. Testing circuit breaker check with mock...")
    cb_result = await health_check.check_circuit_breaker()
    print(f"   Result: {cb_result['status']}")
    print(f"   Healthy: {cb_result['healthy']}")
    assert cb_result['status'] == 'healthy'
    
    print("\n5. Testing system resources check...")
    # Mock psutil import
    with patch.dict('sys.modules', {'psutil': Mock(), 'os': Mock()}):
        mock_psutil = sys.modules['psutil']
        mock_os = sys.modules['os']
        
        # Mock psutil functions
        mock_psutil.cpu_percent = Mock(return_value=50.0)
        mock_psutil.virtual_memory = Mock(return_value=Mock(percent=60.0, available=8*1024**3))
        mock_psutil.disk_usage = Mock(return_value=Mock(percent=70.0, free=100*1024**3))
        mock_psutil.Process = Mock(return_value=Mock(
            memory_info=Mock(return_value=Mock(rss=500*1024**2)),
            cpu_percent=Mock(return_value=10.0)
        ))
        mock_os.getpid = Mock(return_value=1234)
        
        sys_result = await health_check.check_system_resources()
        print(f"   Result: {sys_result['status']}")
        print(f"   Healthy: {sys_result['healthy']}")
        assert sys_result['status'] == 'healthy'
    
    print("\n6. Testing run_all_checks() with mocks...")
    all_results = await health_check.run_all_checks()
    print(f"   Overall status: {all_results['status']}")
    print(f"   Healthy: {all_results['healthy']}")
    print(f"   Number of checks: {len(all_results['checks'])}")
    
    # Verify all checks are present
    expected_checks = ['exchange_connection', 'database', 'ml_model', 'circuit_breaker', 'redis', 'system_resources']
    for check in expected_checks:
        assert check in all_results['checks']
        print(f"   - {check}: {all_results['checks'][check]['status']}")
    
    print("\n✅ All tests passed with mocked dependencies!")
    return 0

async def main():
    """Main test function."""
    print("=" * 60)
    print("Stoic Citadel Health Check - Mock Test")
    print("=" * 60)
    
    try:
        result = await test_health_check_with_mocks()
        print("\n" + "=" * 60)
        print("Test Summary: ✅ ALL TESTS PASSED")
        print("=" * 60)
        return result
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    # Run async tests
    result = asyncio.run(main())
    sys.exit(result)
