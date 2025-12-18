# Testing Guide / Руководство по тестированию

## Quick Start / Быстрый старт

```bash
# 1. Activate virtual environment / Активировать виртуальное окружение
cd C:\Users\Владислав\Documents\GitHub\hft-algotrade-bot
.venv\Scripts\Activate.ps1

# 2. Install dependencies / Установить зависимости
pip install -r requirements.txt

# 3. Run all tests / Запустить все тесты
pytest tests/ -v

# 4. Run with coverage / Запустить с покрытием
pytest tests/ -v --cov=src --cov-report=html
```

---

## Test Categories / Категории тестов

### 1. Unit Tests (174 tests) / Юнит-тесты

```bash
# All unit tests / Все юнит-тесты
pytest tests/ -v

# By module / По модулям:
pytest tests/test_risk/ -v           # Risk management (15 tests)
pytest tests/test_ml/ -v             # ML Pipeline (42 tests)
pytest tests/test_strategies/ -v     # Trading strategies (50 tests)
pytest tests/test_monitoring/ -v     # Monitoring (20 tests)
```

### 2. Fast Tests Only / Только быстрые тесты

```bash
# Skip slow tests / Пропустить медленные тесты
pytest tests/ -v -m "not slow"

# Only fast tests under 1 second / Только быстрые < 1 сек
pytest tests/ -v --timeout=1
```

### 3. Integration Tests / Интеграционные тесты

```bash
# Test strategy with real data / Тест стратегии с реальными данными
freqtrade backtesting --strategy StoicEnsembleStrategy --timeframe 5m --timerange 20240101-20240201
```

---

## Testing Workflow / Рабочий процесс тестирования

### Step 1: Run Unit Tests / Шаг 1: Юнит-тесты

```bash
pytest tests/ -v
```

**Expected output / Ожидаемый результат:**
```
=================== 174 passed in 45.32s ===================
```

### Step 2: Check Coverage / Шаг 2: Проверка покрытия

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

**Target coverage / Целевое покрытие:** >80%

### Step 3: Backtest Strategy / Шаг 3: Бэктест стратегии

```bash
# Basic backtest / Базовый бэктест
freqtrade backtesting \
    --config user_data/config/config_production.json \
    --strategy StoicEnsembleStrategy \
    --timeframe 5m \
    --timerange 20240101-20240601

# With detailed stats / С детальной статистикой
freqtrade backtesting \
    --config user_data/config/config_production.json \
    --strategy StoicEnsembleStrategy \
    --timeframe 5m \
    --timerange 20240101-20240601 \
    --export trades
```

### Step 4: Walk-Forward Analysis / Шаг 4: Walk-Forward анализ

**Критически важно для ML-стратегий!**

```bash
# Month 1: Train on Jan, Test on Feb
freqtrade backtesting --strategy StoicEnsembleStrategy --timerange 20240101-20240131 --export trades
freqtrade backtesting --strategy StoicEnsembleStrategy --timerange 20240201-20240229 --export trades

# Month 2: Train on Feb, Test on Mar
freqtrade backtesting --strategy StoicEnsembleStrategy --timerange 20240201-20240229 --export trades
freqtrade backtesting --strategy StoicEnsembleStrategy --timerange 20240301-20240331 --export trades

# ... repeat for all months
```

### Step 5: Stress Testing / Шаг 5: Стресс-тестирование

```bash
# Test with high slippage / Тест с высоким проскальзыванием
freqtrade backtesting \
    --strategy StoicEnsembleStrategy \
    --timerange 20240101-20240601 \
    --fee 0.002  # 0.2% fees (2x normal)
```

---

## Key Test Files / Ключевые тестовые файлы

| File | Tests | Description |
|------|-------|-------------|
| `tests/test_ml/test_feature_engineering.py` | 15 | Feature generation |
| `tests/test_ml/test_model_trainer.py` | 12 | Model training |
| `tests/test_ml/test_model_registry.py` | 8 | Model versioning |
| `tests/test_risk/test_circuit_breaker.py` | 10 | Circuit breaker |
| `tests/test_risk/test_correlation.py` | 5 | Correlation checks |
| `tests/test_strategies/test_stoic_ensemble.py` | 25 | Main strategy |
| `tests/test_monitoring/test_trading_metrics.py` | 12 | Prometheus metrics |

---

## Common Issues / Частые проблемы

### Issue 1: ModuleNotFoundError

```
ModuleNotFoundError: No module named 'sklearn'
```

**Solution / Решение:**
```bash
pip install scikit-learn>=1.3.0
```

### Issue 2: Import errors in tests

```
ModuleNotFoundError: No module named 'src'
```

**Solution / Решение:**
```bash
# Add project root to PYTHONPATH
$env:PYTHONPATH = "C:\Users\Владислав\Documents\GitHub\hft-algotrade-bot"
pytest tests/ -v
```

### Issue 3: Async test warnings

```
PytestUnraisableExceptionWarning: asyncio...
```

**Solution / Решение:**
```bash
pip install pytest-asyncio>=0.23.0
```

---

## Performance Benchmarks / Бенчмарки производительности

### Expected Test Times / Ожидаемое время

| Test Suite | Expected Time |
|------------|---------------|
| All tests | ~45-60 seconds |
| ML tests only | ~20 seconds |
| Strategy tests | ~15 seconds |
| Risk tests | ~5 seconds |

### Backtest Benchmarks / Бенчмарки бэктеста

| Timerange | Expected Time |
|-----------|---------------|
| 1 month | ~30 seconds |
| 6 months | ~3 minutes |
| 1 year | ~6 minutes |

---

## Validation Checklist / Чек-лист валидации

Before deploying to production / Перед продакшеном:

- [ ] All 174 unit tests pass
- [ ] Coverage > 80%
- [ ] Backtest shows positive Sharpe Ratio
- [ ] Walk-forward analysis profitable for >50% of periods
- [ ] Stress test with 0.2% fees still profitable
- [ ] Circuit breaker triggers correctly at -5% drawdown
- [ ] ML model accuracy > 52% on test set

---

## Docker Testing / Тестирование в Docker

```bash
# Build and run tests in container
docker-compose build
docker-compose run --rm freqtrade pytest tests/ -v

# Run backtest in container
docker-compose run --rm freqtrade backtesting \
    --strategy StoicEnsembleStrategy \
    --timerange 20240101-20240301
```

---

## CI/CD Integration / CI/CD интеграция

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --cov=src
```

---

## Need Help? / Нужна помощь?

1. Check logs: `user_data/logs/freqtrade.log`
2. Run with debug: `pytest tests/ -v -s --tb=long`
3. Open issue: https://github.com/kandibobe/hft-algotrade-bot/issues
