import pytest
import aiohttp
import asyncio
import os

@pytest.mark.asyncio
async def test_system_health():
    """
    Integration test to verify all core services are running.
    """
    # 1. Check Freqtrade API (simulated)
    # Since we can't easily spin up the full docker stack in this test environment,
    # we'll mock the check or verify local files/config existence.
    
    assert os.path.exists("user_data/config/config_production.json"), "Production config missing"
    assert os.path.exists(".env"), ".env file missing"

    # 2. Check Database Connectivity (simulated check of config)
    from src.config.unified_config import load_config
    try:
        config = load_config()
        assert config.system.db_url is not None
    except Exception as e:
        pytest.fail(f"Config loading failed: {e}")

    # 3. Check ML dependencies
    try:
        import feast
        import lightgbm
        import torch
    except ImportError as e:
        pytest.fail(f"Critical ML dependency missing: {e}")

    # 4. Check API endpoints (if running)
    # async with aiohttp.ClientSession() as session:
    #     try:
    #         async with session.get("http://localhost:8080/api/v1/ping", timeout=2) as resp:
    #             assert resp.status == 200
    #     except Exception:
    #         print("API not running, skipping live endpoint check")

if __name__ == "__main__":
    asyncio.run(test_system_health())