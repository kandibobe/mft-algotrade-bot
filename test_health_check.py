#!/usr/bin/env python3
"""
Test script for HealthCheck functionality.
"""

import asyncio
import logging
import sys
from src.monitoring.health_check import HealthCheck

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_health_check():
    """Test the HealthCheck class."""
    print("Testing HealthCheck class...")
    
    try:
        # Create health check instance
        health_check = HealthCheck()
        
        # Test individual checks
        print("\n1. Testing exchange connection check...")
        exchange_result = await health_check.check_exchange()
        print(f"   Result: {exchange_result['status']} - {exchange_result['details']}")
        
        print("\n2. Testing database connection check...")
        db_result = await health_check.check_database()
        print(f"   Result: {db_result['status']} - {db_result['details']}")
        
        print("\n3. Testing ML model check...")
        ml_result = await health_check.check_ml_model()
        print(f"   Result: {ml_result['status']} - {ml_result['details']}")
        
        print("\n4. Testing circuit breaker check...")
        cb_result = await health_check.check_circuit_breaker()
        print(f"   Result: {cb_result['status']} - {cb_result['details']}")
        
        print("\n5. Testing Redis check...")
        redis_result = await health_check.check_redis()
        print(f"   Result: {redis_result['status']} - {redis_result['details']}")
        
        print("\n6. Testing system resources check...")
        sys_result = await health_check.check_system_resources()
        print(f"   Result: {sys_result['status']}")
        if 'details' in sys_result:
            details = sys_result['details']
            print(f"   CPU: {details.get('cpu_percent', 'N/A')}%")
            print(f"   Memory: {details.get('memory_percent', 'N/A')}%")
            print(f"   Disk: {details.get('disk_percent', 'N/A')}%")
        
        # Test running all checks
        print("\n7. Testing run_all_checks()...")
        all_results = await health_check.run_all_checks()
        print(f"   Overall status: {all_results['status']}")
        print(f"   Healthy: {all_results['healthy']}")
        print(f"   Timestamp: {all_results['timestamp']}")
        
        print("\n8. Individual check results:")
        for check_name, result in all_results['checks'].items():
            status = result.get('status', 'unknown')
            healthy = result.get('healthy', False)
            print(f"   - {check_name}: {status} (healthy: {healthy})")
        
        print("\n✅ Health check test completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Health check test failed: {e}", exc_info=True)
        print(f"\n❌ Health check test failed: {e}")
        return 1

async def test_fastapi_availability():
    """Test if FastAPI is available."""
    print("\nTesting FastAPI availability...")
    try:
        from src.monitoring.health_check import FASTAPI_AVAILABLE, app
        if FASTAPI_AVAILABLE:
            print("✅ FastAPI is available")
            if app is not None:
                print("✅ FastAPI app is initialized")
            else:
                print("⚠️ FastAPI app is None")
        else:
            print("⚠️ FastAPI is not available. Install with: pip install fastapi uvicorn")
        return FASTAPI_AVAILABLE
    except Exception as e:
        print(f"❌ Error checking FastAPI availability: {e}")
        return False

async def main():
    """Main test function."""
    print("=" * 60)
    print("Stoic Citadel Health Check System Test")
    print("=" * 60)
    
    # Test health check functionality
    health_check_result = await test_health_check()
    
    # Test FastAPI availability
    fastapi_result = await test_fastapi_availability()
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  Health Check Tests: {'✅ PASSED' if health_check_result == 0 else '❌ FAILED'}")
    print(f"  FastAPI Available: {'✅ YES' if fastapi_result else '⚠️ NO (install with: pip install fastapi uvicorn)'}")
    print("=" * 60)
    
    return health_check_result

if __name__ == "__main__":
    # Run async tests
    result = asyncio.run(main())
    sys.exit(result)
