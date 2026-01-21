# 🧪 Testing Guide - Stoic Citadel Trading Bot

Complete guide to running and understanding the test suite.

---

## 📋 Quick Start / Быстрый старт

```bash
# 1. Activate virtual environment / Активировать виртуальное окружение
cd C:\mft-algotrade-bot
.venv\Scripts\Activate.ps1

# 2. Install dependencies / Установить зависимости
pip install -e ".[dev]"

# 3. Run all tests / Запустить все тесты
pytest tests/ -v

# 4. Run with coverage / Запустить с покрытием
pytest tests/ -v --cov=src --cov-report=html
```

---

## 🎯 Critical Tests (Must Pass!) / Критические тесты

### 1. Data Leakage Test (CRITICAL!)
**Why:** Prevents "too good to be true" backtests that fail in production.

```bash
pytest tests/test_ml/test_data_leakage.py::TestFeatureLeakage::test_vwap_fixed_no_leakage -v
```

---

### 2. Race Condition Test (CRITICAL!)
**Why:** Prevents order state corruption in production.

```bash
pytest tests/test_order_manager/test_async_executor.py::TestRaceConditions::test_order_fills_during_cancel_attempt -v
```

---

### 3. Triple Barrier Correctness (CRITICAL!)
**Why:** Ensures ML labels are correct (garbage labels = garbage model).

```bash
pytest tests/test_ml/test_triple_barrier.py::TestTripleBarrierBasic -v
```

---

## 📊 Test Categories / Категории тестов

### Unit Tests / Юнит-тесты

```bash
# All unit tests / Все юнит-тесты
pytest tests/ -v

# By module / По модулям:
pytest tests/test_risk/ -v           # Risk management
pytest tests/test_ml/ -v             # ML Pipeline
pytest tests/test_strategies/ -v     # Trading strategies
```

### Integration Tests / Интеграционные тесты

```bash
# Test strategy with real data / Тест стратегии с реальными данными
freqtrade backtesting --strategy StoicEnsembleStrategy --timeframe 5m --timerange 20240101-20240201
```

---

## 🔬 Advanced Testing / Продвинутое тестирование

### Run with Coverage
```bash
# Generate HTML coverage report
pytest tests/ --cov=src --cov-report=html
```

**Target Coverage:** > 80%

### Load Testing with Locust / Нагрузочное тестирование
```bash
pip install locust
locust -f tests/load_test.py --host http://localhost:8080
```

### Docker Testing / Тестирование в Docker
```bash
# Build and run tests in container
docker-compose -f deploy/docker-compose.test.yml build
docker-compose -f deploy/docker-compose.test.yml run --rm freqtrade pytest tests/ -v
```

---

## 🎯 Pre-Deployment Checklist / Чек-лист перед деплоем

- [ ] All unit tests pass / Все юнит-тесты пройдены
- [ ] Coverage > 80% / Покрытие > 80%
- [ ] Backtest shows positive Sharpe Ratio / Бэктест показывает положительный коэф. Шарпа
- [ ] Circuit breaker triggers correctly / Circuit breaker срабатывает корректно
- [ ] ML model accuracy > 52% on test set / Точность ML модели > 52%

---

**Happy Testing! 🧪**
