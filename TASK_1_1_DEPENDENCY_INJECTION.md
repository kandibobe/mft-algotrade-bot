# Задача 1.1: Внедрение Dependency Injection

## Контекст
Текущая архитектура Stoic Citadel использует прямые инстанцирования зависимостей, что приводит к:
1. Высокой связности между компонентами
2. Сложности тестирования (невозможно легко мокать зависимости)
3. Трудностям при замене реализаций

## Цель
Внедрить контейнер зависимостей для управления инстанцированием и внедрением зависимостей.

## Требования
1. **Обратная совместимость**: Существующий код должен продолжать работать
2. **Минимальные изменения**: Изменения только в точках инстанцирования
3. **Поддержка async**: DI должен работать с асинхронными зависимостями
4. **Тестируемость**: Упрощение мокирования зависимостей в тестах

## Детальный план реализации

### Шаг 1: Создание контейнера DI
**Файл**: `src/core/di.py`

```python
"""
Dependency Injection Container for Stoic Citadel.

Features:
- Async dependency support
- Singleton and transient scopes
- Type hints support
- Testing utilities
"""

from typing import Any, Type, TypeVar, Callable, Optional
from functools import wraps
import inspect
import logging

T = TypeVar('T')

logger = logging.getLogger(__name__)


class DIContainer:
    """Simple DI container with async support."""
    
    def __init__(self):
        self._singletons = {}
        self._factories = {}
        self._scoped = {}
    
    def register_singleton(self, interface: Type, implementation: Any):
        """Register singleton implementation."""
        self._singletons[interface] = implementation
        logger.debug(f"Registered singleton: {interface.__name__} -> {implementation.__class__.__name__}")
    
    def register_factory(self, interface: Type, factory: Callable):
        """Register factory for transient dependencies."""
        self._factories[interface] = factory
        logger.debug(f"Registered factory: {interface.__name__}")
    
    async def resolve(self, interface: Type) -> Any:
        """Resolve dependency."""
        # Check singletons first
        if interface in self._singletons:
            return self._singletons[interface]
        
        # Check factories
        if interface in self._factories:
            factory = self._factories[interface]
            if inspect.iscoroutinefunction(factory):
                return await factory()
            return factory()
        
        # Try to instantiate directly
        try:
            return interface()
        except Exception as e:
            raise ValueError(f"Cannot resolve dependency: {interface.__name__}") from e
    
    def clear(self):
        """Clear all registrations (for testing)."""
        self._singletons.clear()
        self._factories.clear()
        self._scoped.clear()


# Global container instance
_container = DIContainer()


def get_container() -> DIContainer:
    """Get global DI container."""
    return _container


def inject(*dependencies):
    """
    Decorator to inject dependencies into function.
    
    Usage:
        @inject(ExchangeAPI, RedisClient)
        async def my_function(exchange: ExchangeAPI, redis: RedisClient):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            container = get_container()
            
            # Resolve dependencies
            resolved_deps = []
            for dep in dependencies:
                resolved = await container.resolve(dep)
                resolved_deps.append(resolved)
            
            # Call function with injected dependencies
            return await func(*args, *resolved_deps, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            container = get_container()
            
            # Resolve dependencies
            resolved_deps = []
            for dep in dependencies:
                resolved = container.resolve(dep)
                if inspect.isawaitable(resolved):
                    raise RuntimeError(f"Dependency {dep} is async, use @inject with async function")
                resolved_deps.append(resolved)
            
            # Call function with injected dependencies
            return func(*args, *resolved_deps, **kwargs)
        
        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


class ServiceProvider:
    """Context manager for scoped services."""
    
    def __init__(self, container: Optional[DIContainer] = None):
        self.container = container or DIContainer()
        self._parent = None
    
    async def __aenter__(self):
        # Store parent container
        global _container
        self._parent = _container
        _container = self.container
        return self.container
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Restore parent container
        global _container
        _container = self._parent
    
    def get_service(self, interface: Type) -> Any:
        """Get service from container (sync)."""
        return self.container.resolve(interface)
```

### Шаг 2: Создание фабрик для ключевых сервисов
**Файл**: `src/core/factories.py`

```python
"""
Factory functions for DI container.
"""

import logging
from typing import Dict, Any

from src.ml.inference_service import MLInferenceService, MLModelConfig
from src.websocket.data_stream import WebSocketDataStream, StreamConfig
from src.order_manager.order_executor import OrderExecutor, ExecutionMode
from src.order_manager.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


async def create_ml_inference_service() -> MLInferenceService:
    """Factory for ML inference service."""
    from src.ml.redis_client import get_redis_client
    
    redis_client = await get_redis_client()
    
    # Load model configurations
    models = {
        "trend_classifier": MLModelConfig(
            model_name="trend_classifier",
            model_path="user_data/models/trend_classifier.pkl",
            feature_columns=["rsi", "macd", "bb_width", "volume_ratio"],
            prediction_threshold=0.6,
            timeout_ms=100
        )
    }
    
    service = MLInferenceService(
        redis_url="redis://localhost:6379",
        models=models
    )
    await service.start()
    
    logger.info("ML Inference Service created via factory")
    return service


async def create_websocket_stream(config: StreamConfig) -> WebSocketDataStream:
    """Factory for WebSocket stream."""
    stream = WebSocketDataStream(config)
    # Note: stream.start() should be called separately
    return stream


def create_order_executor(
    mode: ExecutionMode = ExecutionMode.PAPER,
    circuit_breaker: Optional[CircuitBreaker] = None
) -> OrderExecutor:
    """Factory for order executor."""
    from src.order_manager.slippage_simulator import SlippageSimulator
    
    executor = OrderExecutor(
        mode=mode,
        circuit_breaker=circuit_breaker,
        slippage_simulator=SlippageSimulator(),
        max_retries=3,
        retry_delay_ms=100
    )
    
    logger.info(f"Order Executor created in {mode.value} mode")
    return executor


def create_circuit_breaker(
    max_errors: int = 10,
    reset_timeout: int = 300
) -> CircuitBreaker:
    """Factory for circuit breaker."""
    return CircuitBreaker(
        max_errors=max_errors,
        reset_timeout=reset_timeout
    )
```

### Шаг 3: Регистрация зависимостей
**Файл**: `src/core/bootstrap.py`

```python
"""
Bootstrap DI container with all dependencies.
"""

import logging
from typing import Dict, Any

from src.core.di import get_container, DIContainer
from src.core.factories import (
    create_ml_inference_service,
    create_order_executor,
    create_circuit_breaker,
    create_websocket_stream
)
from src.websocket.data_stream import StreamConfig, Exchange
from src.order_manager.order_executor import ExecutionMode

logger = logging.getLogger(__name__)


async def bootstrap_container(
    config: Dict[str, Any] = None
) -> DIContainer:
    """
    Bootstrap DI container with all dependencies.
    
    Args:
        config: Application configuration
        
    Returns:
        Configured DI container
    """
    container = get_container()
    config = config or {}
    
    # Register core services
    container.register_factory(
        MLInferenceService,
        create_ml_inference_service
    )
    
    container.register_factory(
        OrderExecutor,
        lambda: create_order_executor(
            mode=ExecutionMode(config.get("execution_mode", "paper")),
            circuit_breaker=container.resolve(CircuitBreaker)
        )
    )
    
    container.register_factory(
        CircuitBreaker,
        lambda: create_circuit_breaker(
            max_errors=config.get("circuit_breaker_max_errors", 10),
            reset_timeout=config.get("circuit_breaker_reset_timeout", 300)
        )
    )
    
    # Register WebSocket stream with configuration
    stream_config = StreamConfig(
        exchange=Exchange.BINANCE,
        symbols=config.get("symbols", ["BTC/USDT", "ETH/USDT"]),
        channels=["ticker", "trade"]
    )
    container.register_factory(
        WebSocketDataStream,
        lambda: create_websocket_stream(stream_config)
    )
    
    logger.info("DI container bootstrapped successfully")
    return container


async def shutdown_container():
    """Shutdown all services in container."""
    container = get_container()
    
    # Cleanup resources
    # Note: In production, you would iterate through services and call .stop()
    
    container.clear()
    logger.info("DI container shutdown complete")
```

### Шаг 4: Рефакторинг OrderExecutor
**Файл**: `src/order_manager/order_executor.py`

Изменения:
1. Заменить прямые инстанцирования на DI
2. Добавить поддержку внедрения зависимостей

```python
# В начале файла добавить импорт
from src.core.di import inject

# Изменить конструктор OrderExecutor
class OrderExecutor:
    @inject(CircuitBreaker, SlippageSimulator)
    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.LIVE,
        circuit_breaker: Optional[CircuitBreaker] = None,
        slippage_simulator: Optional[SlippageSimulator] = None,
        max_retries: int = 3,
        retry_delay_ms: int = 100,
    ):
        # Остальной код без изменений
        ...
```

### Шаг 5: Рефакторинг MLInferenceService
**Файл**: `src/ml/inference_service.py`

```python
# В начале файла добавить импорт
from src.core.di import inject

# Изменить конструктор MLInferenceService
class MLInferenceService:
    @inject()  # Пока без зависимостей, но готово для будущего расширения
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        models: Optional[Dict[str, MLModelConfig]] = None
    ):
        # Остальной код без изменений
        ...
```

### Шаг 6: Рефакторинг WebSocketDataStream
**Файл**: `src/websocket/data_stream.py`

```python
# В начале файла добавить импорт
from src.core.di import inject

# Изменить конструктор WebSocketDataStream
class WebSocketDataStream:
    @inject()  # Пока без зависимостей
    def __init__(self, config: StreamConfig):
        # Остальной код без изменений
        ...
```

### Шаг 7: Обновление тестов
**Файл**: `tests/test_order_manager/test_order_executor.py`

```python
import pytest
from unittest.mock import Mock, AsyncMock
from src.core.di import DIContainer
from src.order_manager.order_executor import OrderExecutor, ExecutionMode


@pytest.fixture
async def di_container():
    """Test DI container."""
    container = DIContainer()
    
    # Register mocks
    container.register_singleton(CircuitBreaker, Mock())
    container.register_singleton(SlippageSimulator, Mock())
    
    yield container


@pytest.mark.asyncio
async def test_order_executor_with_di(di_container):
    """Test OrderExecutor with DI."""
    # Use container to resolve dependencies
    executor = await di_container.resolve(OrderExecutor)
    
    assert executor is not None
    assert executor.mode == ExecutionMode.LIVE  # Default
```

### Шаг 8: Создание примеров использования
**Файл**: `examples/di_usage_example.py`

```python
"""
Example of using DI container in Stoic Citadel.
"""

import asyncio
import logging
from src.core.bootstrap import bootstrap_container, shutdown_container
from src.core.di import get_container
from src.order_manager.order_executor import OrderExecutor
from src.ml.inference_service import MLInferenceService

logging.basicConfig(level=logging.INFO)


async def main():
    """Example usage of DI container."""
    
    # Bootstrap container with configuration
    config = {
        "execution_mode": "paper",
        "symbols": ["BTC/USDT", "ETH/USDT"],
        "circuit_breaker_max_errors": 5
    }
    
    await bootstrap_container(config)
    container = get_container()
    
    try:
        # Resolve and use services
        executor = await container.resolve(OrderExecutor)
        ml_service = await container.resolve(MLInferenceService)
        
        print(f"Order Executor mode: {executor.mode}")
        print(f"ML Service stats: {ml_service.get_stats()}")
        
        # Use services...
        
    finally:
        await shutdown_container()


if __name__ == "__main__":
    asyncio.run(main())
```

## Тестирование

### Unit тесты
1. **Тест контейнера DI**: Проверка регистрации и разрешения зависимостей
2. **Тест фабрик**: Проверка создания сервисов
3. **Тест декоратора @inject**: Проверка внедрения зависимостей

### Интеграционные тесты
1. **Тест bootstrap**: Проверка инициализации контейнера
2. **Тест жизненного цикла**: Проверка startup/shutdown
3. **Тест совместимости**: Проверка работы с существующим кодом

## Валидация

### Критерии успеха
1. ✅ Существующий код работает без изменений
2. ✅ Новый код использует DI
3. ✅ Тесты проходят
4. ✅ Производительность не ухудшилась
5. ✅ Логирование DI операций

### Метрики
1. **Coverage**: 90%+ для новых файлов DI
2. **Performance**: <1ms overhead на разрешение зависимости
3. **Memory**: Нет утечек памяти

## Риски и митигации

### Риск 1: Нарушение обратной совместимости
**Митигация**: 
- Сохранить старые конструкторы с параметрами по умолчанию
- Использовать @inject как опциональный декоратор
- Поэтапный rollout

### Риск 2: Сложность отладки
**Митигация**:
- Подробное логирование разрешения зависимостей
- Валидация конфигурации при bootstrap
- Clear error messages

### Риск 3: Производительность
**Митигация**:
- Кэширование разрешенных зависимостей
- Профилирование критических путей
- Async-first дизайн

## Следующие шаги после завершения

1. **Документация**: Обновить README с примерами DI
2. **Обучение**: Провести code review и обучение команды
3. **Мониторинг**: Добавить метрики использования DI
4. **Оптимизация**: Профилирование и оптимизация при необходимости

## Время выполнения
**Оценка**: 2-3 дня для опытного разработчика

**Разбивка**:
- День 1: Создание DI контейнера и фабрик
- День 2: Рефакторинг ключевых сервисов
- День 3: Тестирование и документация

Эта задача самодостаточна и может быть выполнена независимо от других задач.
