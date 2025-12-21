"""
Stoic Citadel - Health Check System
====================================

Comprehensive health checks for Kubernetes orchestration and monitoring.
Provides liveness and readiness probes for container orchestration.

Features:
- Exchange connection health check
- Database connection health check
- ML model inference health check
- Circuit breaker status check
- FastAPI endpoints for Kubernetes probes

Usage:
    # Run as standalone service
    uvicorn src.monitoring.health_check:app --host 0.0.0.0 --port 8080

    # Or integrate with main bot
    from src.monitoring.health_check import HealthCheck, app
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

# Try to import FastAPI
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available. Install with: pip install fastapi uvicorn")

# Try to import existing bot components
try:
    from src.monitoring.metrics_exporter import get_exporter

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

try:
    from config.database import get_session

    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

try:
    from src.risk.circuit_breaker import CircuitBreaker

    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False

try:
    from src.ml.inference_service import MLInferenceService

    ML_INFERENCE_AVAILABLE = True
except ImportError:
    ML_INFERENCE_AVAILABLE = False

logger = logging.getLogger(__name__)


class HealthCheck:
    """
    Comprehensive health check system for trading bot components.

    Provides async health checks for:
    - Exchange API connectivity
    - Database connection
    - ML model inference
    - Circuit breaker status
    - Redis connection (if used)
    - System resources
    """

    def __init__(self, bot=None):
        """
        Initialize health check system.

        Args:
            bot: Optional bot instance for component access
        """
        self.bot = bot
        self.checks = {
            "exchange_connection": self.check_exchange,
            "database": self.check_database,
            "ml_model": self.check_ml_model,
            "circuit_breaker": self.check_circuit_breaker,
            "redis": self.check_redis,
            "system_resources": self.check_system_resources,
        }

        # Initialize component clients
        self._init_clients()

        logger.info("Health check system initialized")

    def _init_clients(self):
        """Initialize component clients for health checks."""
        # Exchange client
        self.exchange_client = None
        if hasattr(self.bot, "exchange"):
            self.exchange_client = self.bot.exchange
        else:
            # Try to create exchange client
            try:
                import ccxt

                # Use a simple exchange client for health check
                # In production, this would use the bot's actual exchange client
                self.exchange_client = ccxt.binance(
                    {
                        "enableRateLimit": True,
                        "timeout": 5000,
                    }
                )
            except ImportError:
                logger.warning("ccxt not available for exchange health checks")

        # Database client
        self.db_client = None
        if DATABASE_AVAILABLE:
            try:
                self.db_client = get_session
            except Exception as e:
                logger.warning(f"Failed to initialize database client: {e}")

        # ML inference client
        self.ml_client = None
        if ML_INFERENCE_AVAILABLE and hasattr(self.bot, "ml_inference_service"):
            self.ml_client = self.bot.ml_inference_service
        elif ML_INFERENCE_AVAILABLE:
            try:
                self.ml_client = MLInferenceService()
            except Exception as e:
                logger.warning(f"Failed to initialize ML inference client: {e}")

        # Circuit breaker
        self.circuit_breaker = None
        if CIRCUIT_BREAKER_AVAILABLE and hasattr(self.bot, "circuit_breaker"):
            self.circuit_breaker = self.bot.circuit_breaker
        elif CIRCUIT_BREAKER_AVAILABLE:
            try:
                self.circuit_breaker = CircuitBreaker()
            except Exception as e:
                logger.warning(f"Failed to initialize circuit breaker: {e}")

        # Redis client
        self.redis_client = None
        try:
            import redis.asyncio as redis

            self.redis_client = redis.from_url("redis://localhost:6379")
        except ImportError:
            logger.warning("redis not available for health checks")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis client: {e}")

    async def check_exchange(self) -> Dict[str, Any]:
        """
        Check if exchange API is reachable and responsive.

        Returns:
            Dict with status and details
        """
        if not self.exchange_client:
            return {
                "status": "unknown",
                "details": "Exchange client not available",
                "healthy": False,
            }

        try:
            # Try to fetch a ticker with timeout
            start_time = datetime.utcnow()

            # Use asyncio timeout for async exchange clients
            if hasattr(self.exchange_client, "fetch_ticker"):
                if asyncio.iscoroutinefunction(self.exchange_client.fetch_ticker):
                    ticker = await asyncio.wait_for(
                        self.exchange_client.fetch_ticker("BTC/USDT"), timeout=5.0
                    )
                else:
                    # Sync version
                    ticker = self.exchange_client.fetch_ticker("BTC/USDT")

                latency = (datetime.utcnow() - start_time).total_seconds()

                return {
                    "status": "healthy",
                    "details": {
                        "exchange": getattr(self.exchange_client, "name", "unknown"),
                        "symbol": "BTC/USDT",
                        "last_price": ticker.get("last"),
                        "latency_seconds": round(latency, 3),
                    },
                    "healthy": True,
                }
            else:
                # Simple ping check
                if hasattr(self.exchange_client, "ping"):
                    if asyncio.iscoroutinefunction(self.exchange_client.ping):
                        await asyncio.wait_for(self.exchange_client.ping(), timeout=5.0)
                    else:
                        self.exchange_client.ping()

                    return {
                        "status": "healthy",
                        "details": "Exchange ping successful",
                        "healthy": True,
                    }
                else:
                    return {
                        "status": "unknown",
                        "details": "Exchange client doesn't support health checks",
                        "healthy": False,
                    }

        except asyncio.TimeoutError:
            return {"status": "unhealthy", "details": "Exchange API timeout (5s)", "healthy": False}
        except Exception as e:
            logger.error(f"Exchange health check failed: {e}")
            return {"status": "unhealthy", "details": f"Exchange error: {str(e)}", "healthy": False}

    async def check_database(self) -> Dict[str, Any]:
        """
        Check database connection and basic query execution.

        Returns:
            Dict with status and details
        """
        if not self.db_client:
            return {
                "status": "unknown",
                "details": "Database client not available",
                "healthy": False,
            }

        try:
            start_time = datetime.utcnow()

            # Execute a simple query
            if hasattr(self.db_client, "__call__"):
                # It's a session context manager
                async with self.db_client() as session:
                    result = await session.execute("SELECT 1")
                    test_value = result.scalar()
            else:
                # Try sync version
                result = self.db_client.execute("SELECT 1")
                test_value = result.scalar()

            latency = (datetime.utcnow() - start_time).total_seconds()

            if test_value == 1:
                return {
                    "status": "healthy",
                    "details": {
                        "query": "SELECT 1",
                        "result": test_value,
                        "latency_seconds": round(latency, 3),
                    },
                    "healthy": True,
                }
            else:
                return {
                    "status": "unhealthy",
                    "details": f"Unexpected query result: {test_value}",
                    "healthy": False,
                }

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {"status": "unhealthy", "details": f"Database error: {str(e)}", "healthy": False}

    async def check_ml_model(self) -> Dict[str, Any]:
        """
        Check if ML model can make predictions.

        Returns:
            Dict with status and details
        """
        if not self.ml_client:
            return {
                "status": "unknown",
                "details": "ML inference client not available",
                "healthy": False,
            }

        try:
            start_time = datetime.utcnow()

            # Create sample features for health check
            sample_features = {}

            # Try to get feature columns from model config
            if hasattr(self.ml_client, "models"):
                for model_name, model_config in self.ml_client.models.items():
                    if hasattr(model_config, "feature_columns"):
                        for col in model_config.feature_columns[:5]:  # First 5 columns
                            sample_features[col] = 0.5
                        break

            # If no feature columns found, use default
            if not sample_features:
                sample_features = {
                    "rsi": 50.0,
                    "macd": 0.0,
                    "volume": 1000.0,
                    "close": 50000.0,
                    "volatility": 0.02,
                }

            # Try to make a prediction
            if hasattr(self.ml_client, "predict"):
                # Get first model name
                model_names = list(getattr(self.ml_client, "models", {}).keys())
                if model_names:
                    model_name = model_names[0]

                    if asyncio.iscoroutinefunction(self.ml_client.predict):
                        result = await self.ml_client.predict(model_name, sample_features)
                    else:
                        result = self.ml_client.predict(model_name, sample_features)

                    latency = (datetime.utcnow() - start_time).total_seconds()

                    return {
                        "status": "healthy",
                        "details": {
                            "model": model_name,
                            "prediction": getattr(result, "prediction", "unknown"),
                            "signal": getattr(result, "signal", "unknown"),
                            "confidence": getattr(result, "confidence", 0.0),
                            "latency_seconds": round(latency, 3),
                        },
                        "healthy": True,
                    }

            # Fallback: just check if client is responsive
            if hasattr(self.ml_client, "health_check"):
                if asyncio.iscoroutinefunction(self.ml_client.health_check):
                    health = await self.ml_client.health_check()
                else:
                    health = self.ml_client.health_check()

                return {
                    "status": "healthy" if health.get("status") == "healthy" else "unhealthy",
                    "details": health,
                    "healthy": health.get("status") == "healthy",
                }

            # If we get here, ML client exists but we can't test it
            return {
                "status": "unknown",
                "details": "ML client available but health check not implemented",
                "healthy": False,
            }

        except Exception as e:
            logger.error(f"ML model health check failed: {e}")
            return {"status": "unhealthy", "details": f"ML model error: {str(e)}", "healthy": False}

    async def check_circuit_breaker(self) -> Dict[str, Any]:
        """
        Check circuit breaker status.

        Returns:
            Dict with status and details
        """
        if not self.circuit_breaker:
            return {
                "status": "unknown",
                "details": "Circuit breaker not available",
                "healthy": True,  # Not unhealthy, just not available
            }

        try:
            # Get circuit breaker status
            if hasattr(self.circuit_breaker, "get_status"):
                status = self.circuit_breaker.get_status()

                # Circuit breaker is healthy if it's not in OPEN state
                # (OPEN means trading is halted, which is a protective state)
                is_healthy = status.get("state") != "open"

                return {
                    "status": "healthy" if is_healthy else "warning",
                    "details": status,
                    "healthy": is_healthy,
                }
            else:
                # Simple check
                if hasattr(self.circuit_breaker, "can_trade"):
                    can_trade = self.circuit_breaker.can_trade()
                    return {
                        "status": "healthy" if can_trade else "warning",
                        "details": {"can_trade": can_trade},
                        "healthy": can_trade,
                    }
                else:
                    return {
                        "status": "unknown",
                        "details": "Circuit breaker doesn't support status checks",
                        "healthy": True,  # Assume healthy if we can't check
                    }

        except Exception as e:
            logger.error(f"Circuit breaker health check failed: {e}")
            return {
                "status": "unhealthy",
                "details": f"Circuit breaker error: {str(e)}",
                "healthy": False,
            }

    async def check_redis(self) -> Dict[str, Any]:
        """
        Check Redis connection.

        Returns:
            Dict with status and details
        """
        if not self.redis_client:
            return {
                "status": "unknown",
                "details": "Redis client not available",
                "healthy": True,  # Redis might be optional
            }

        try:
            start_time = datetime.utcnow()

            if asyncio.iscoroutinefunction(self.redis_client.ping):
                await self.redis_client.ping()
            else:
                self.redis_client.ping()

            latency = (datetime.utcnow() - start_time).total_seconds()

            return {
                "status": "healthy",
                "details": {"latency_seconds": round(latency, 3)},
                "healthy": True,
            }

        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {"status": "unhealthy", "details": f"Redis error: {str(e)}", "healthy": False}

    async def check_system_resources(self) -> Dict[str, Any]:
        """
        Check system resources (CPU, memory, disk).

        Returns:
            Dict with status and details
        """
        try:
            import os

            import psutil

            details = {}

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            details["cpu_percent"] = cpu_percent

            # Memory usage
            memory = psutil.virtual_memory()
            details["memory_percent"] = memory.percent
            details["memory_available_gb"] = round(memory.available / (1024**3), 2)

            # Disk usage (current directory)
            disk = psutil.disk_usage(".")
            details["disk_percent"] = disk.percent
            details["disk_free_gb"] = round(disk.free / (1024**3), 2)

            # Process info
            process = psutil.Process(os.getpid())
            details["process_memory_mb"] = round(process.memory_info().rss / (1024**2), 2)
            details["process_cpu_percent"] = process.cpu_percent(interval=0.1)

            # Determine health status
            # Warning if CPU > 80% or memory > 90% or disk > 95%
            is_healthy = True
            status = "healthy"

            if cpu_percent > 80:
                status = "warning"
                is_healthy = False
            if memory.percent > 90:
                status = "warning"
                is_healthy = False
            if disk.percent > 95:
                status = "warning"
                is_healthy = False

            return {"status": status, "details": details, "healthy": is_healthy}

        except ImportError:
            return {
                "status": "unknown",
                "details": "psutil not available for system checks",
                "healthy": True,  # Not a critical failure
            }
        except Exception as e:
            logger.error(f"System resources health check failed: {e}")
            return {
                "status": "unhealthy",
                "details": f"System check error: {str(e)}",
                "healthy": False,
            }

    async def run_all_checks(self) -> Dict[str, Any]:
        """
        Run all health checks concurrently.

        Returns:
            Dict with overall status and individual check results
        """
        # Run checks concurrently
        check_tasks = []
        check_names = []

        for name, check_fn in self.checks.items():
            check_tasks.append(check_fn())
            check_names.append(name)

        # Wait for all checks to complete
        results = await asyncio.gather(*check_tasks, return_exceptions=True)

        # Process results
        check_results = {}
        all_healthy = True
        any_warning = False

        for name, result in zip(check_names, results):
            if isinstance(result, Exception):
                check_results[name] = {
                    "status": "error",
                    "details": f"Check failed with exception: {str(result)}",
                    "healthy": False,
                }
                all_healthy = False
            else:
                check_results[name] = result
                if not result.get("healthy", False):
                    all_healthy = False
                if result.get("status") == "warning":
                    any_warning = True

        # Determine overall status
        if all_healthy:
            overall_status = "healthy"
        elif any_warning:
            overall_status = "warning"
        else:
            overall_status = "unhealthy"

        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": check_results,
            "healthy": all_healthy,
        }


# FastAPI Application
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Stoic Citadel Health Check API",
        description="Health check endpoints for Kubernetes orchestration",
        version="1.0.0",
    )

    # Global health check instance
    _health_check = HealthCheck()

    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Stoic Citadel Health Check API",
            "version": "1.0.0",
            "endpoints": {
                "/health": "Liveness probe - is process alive?",
                "/ready": "Readiness probe - is service ready to accept traffic?",
                "/health/detailed": "Detailed health check of all components",
            },
        }

    @app.get("/health")
    async def health():
        """
        Liveness probe - is process alive?

        Kubernetes uses this to determine if the container should be restarted.
        Returns 200 OK if the process is alive.
        """
        return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}

    @app.get("/ready")
    async def readiness():
        """
        Readiness probe - is service ready to accept traffic?

        Kubernetes uses this to determine if the container can receive traffic.
        Returns 200 OK if all critical components are healthy.
        Returns 503 Service Unavailable if any critical component is unhealthy.
        """
        try:
            health_check = HealthCheck()
            results = await health_check.run_all_checks()

            if results["healthy"]:
                return JSONResponse(
                    content={
                        "status": "ready",
                        "timestamp": datetime.utcnow().isoformat(),
                        "checks": results["checks"],
                    },
                    status_code=200,
                )
            else:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "status": "not_ready",
                        "timestamp": datetime.utcnow().isoformat(),
                        "checks": results["checks"],
                    },
                )
        except Exception as e:
            logger.error(f"Readiness check failed: {e}")
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "error",
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": str(e),
                },
            )

    @app.get("/health/detailed")
    async def detailed_health():
        """
        Detailed health check of all components.

        Returns comprehensive health status of all components
        without affecting readiness probe status.
        """
        try:
            health_check = HealthCheck()
            results = await health_check.run_all_checks()

            return JSONResponse(content=results, status_code=200)
        except Exception as e:
            logger.error(f"Detailed health check failed: {e}")
            raise HTTPException(
                status_code=500,
                detail={
                    "status": "error",
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": str(e),
                },
            )

    @app.get("/health/{component}")
    async def component_health(component: str):
        """
        Health check for a specific component.

        Args:
            component: Component name (exchange_connection, database, ml_model, circuit_breaker, redis, system_resources)
        """
        health_check = HealthCheck()

        if component not in health_check.checks:
            raise HTTPException(
                status_code=404,
                detail=f"Component '{component}' not found. Available components: {list(health_check.checks.keys())}",
            )

        try:
            check_fn = health_check.checks[component]
            result = await check_fn()

            return JSONResponse(
                content=result, status_code=200 if result.get("healthy", False) else 503
            )
        except Exception as e:
            logger.error(f"Component health check failed for {component}: {e}")
            raise HTTPException(
                status_code=500,
                detail={
                    "status": "error",
                    "component": component,
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": str(e),
                },
            )

else:
    # Create a dummy app if FastAPI is not available
    app = None

    logger.warning(
        "FastAPI is not available. Health check endpoints will not be available. "
        "Install with: pip install fastapi uvicorn"
    )


# Standalone execution
if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Stoic Citadel Health Check Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    if not FASTAPI_AVAILABLE:
        print("Error: FastAPI is not installed.")
        print("Install it with: pip install fastapi uvicorn")
        exit(1)

    print(f"Starting Stoic Citadel Health Check Service on {args.host}:{args.port}")
    print("Endpoints:")
    print(f"  http://{args.host}:{args.port}/health      - Liveness probe")
    print(f"  http://{args.host}:{args.port}/ready       - Readiness probe")
    print(f"  http://{args.host}:{args.port}/health/detailed - Detailed health")

    uvicorn.run(
        "src.monitoring.health_check:app", host=args.host, port=args.port, reload=args.reload
    )
