# 🧪 Testing Guide - Stoic Citadel Trading Bot

Complete guide to running and understanding the test suite.

---

## 📋 Quick Start

### Install Test Dependencies
```bash
pip install pytest pytest-asyncio pandas numpy tenacity pandas-ta
```

### Run All Tests
```bash
# Full test suite (53+ tests)
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html

# Parallel execution (faster)
pytest tests/ -n auto
```

### Run Specific Tests
```bash
# Single test module
pytest tests/test_ml/test_triple_barrier.py -v

# Single test class
pytest tests/test_ml/test_triple_barrier.py::TestTripleBarrierBasic -v

# Single test function
pytest tests/test_ml/test_data_leakage.py::TestFeatureLeakage::test_vwap_fixed_no_leakage -v
```

---

## 🎯 Critical Tests (Must Pass!)

### 1. Data Leakage Test (CRITICAL!)
**Why:** Prevents "too good to be true" backtests that fail in production.

```bash
pytest tests/test_ml/test_data_leakage.py::TestFeatureLeakage::test_vwap_fixed_no_leakage -v
```

**What it tests:**
- VWAP calculation doesn't use future data (cumsum bug fixed)
- Modifies future data and verifies past features unchanged

**Expected output:**
```
tests/test_ml/test_data_leakage.py::TestFeatureLeakage::test_vwap_fixed_no_leakage PASSED [100%]
```

**If it fails:**
- ❌ DO NOT deploy to production
- ❌ Your backtests are lying to you
- ✅ Check feature_engineering.py for cumsum() usage

---

### 2. Race Condition Test (CRITICAL!)
**Why:** Prevents order state corruption in production.

```bash
pytest tests/test_order_manager/test_async_executor.py::TestRaceConditions::test_order_fills_during_cancel_attempt -v
```

**What it tests:**
- Order fills between status check and cancel
- Lock prevents concurrent cancel attempts
- Double-check pattern catches filled orders

**Expected output:**
```
tests/test_order_manager/test_async_executor.py::TestRaceConditions::test_order_fills_during_cancel_attempt PASSED [100%]
```

**If it fails:**
- ❌ Race condition still exists
- ✅ Check AsyncSmartLimitExecutor for lock usage

---

### 3. Triple Barrier Correctness (CRITICAL!)
**Why:** Ensures ML labels are correct (garbage labels = garbage model).

```bash
pytest tests/test_ml/test_triple_barrier.py::TestTripleBarrierBasic -v
```

**What it tests:**
- TP hit first → label = 1
- SL hit first → label = -1
- Time barrier → label = 0
- Both hit → use close price
- Fee adjustment correctness

**Expected output:**
```
tests/test_ml/test_triple_barrier.py::TestTripleBarrierBasic::test_take_profit_hit_first PASSED
tests/test_ml/test_triple_barrier.py::TestTripleBarrierBasic::test_stop_loss_hit_first PASSED
tests/test_ml/test_triple_barrier.py::TestTripleBarrierBasic::test_time_barrier_hit PASSED
tests/test_ml/test_triple_barrier.py::TestTripleBarrierBasic::test_both_barriers_hit_same_candle PASSED
```

---

## 📊 Test Categories

### ML Pipeline Tests (`tests/test_ml/`)

#### `test_triple_barrier.py` (20+ tests)
**Purpose:** Verify Triple Barrier labeling correctness

**Test Classes:**
- `TestTripleBarrierBasic` - Core labeling logic
- `TestTripleBarrierFees` - Fee adjustment correctness
- `TestTripleBarrierEdgeCases` - Boundary conditions
- `TestTripleBarrierMetadata` - Metadata accuracy
- `TestDynamicBarrierLabeler` - ATR-based barriers
- `TestDataLeakagePrevention` - No future data usage

**Key Tests:**
```bash
# Verify TP detection
pytest tests/test_ml/test_triple_barrier.py::TestTripleBarrierBasic::test_take_profit_hit_first -v

# Verify SL detection
pytest tests/test_ml/test_triple_barrier.py::TestTripleBarrierBasic::test_stop_loss_hit_first -v

# Edge case: both barriers hit
pytest tests/test_ml/test_triple_barrier.py::TestTripleBarrierBasic::test_both_barriers_hit_same_candle -v

# Fee adjustment
pytest tests/test_ml/test_triple_barrier.py::TestTripleBarrierFees::test_fee_adjustment_prevents_false_positive -v
```

---

#### `test_data_leakage.py` (15+ tests)
**Purpose:** Ensure no future data leaks into training

**Test Classes:**
- `TestFeatureLeakage` - Feature engineering leakage
- `TestScalerLeakage` - Scaler doesn't see test data
- `TestWalkForwardValidation` - Temporal validation correctness
- `TestLabelingLeakage` - Labels don't use future data
- `TestFeatureCorrelationFilter` - Correlation on train only
- `TestInformationLeakageEdgeCases` - Subtle leakage bugs

**Key Tests:**
```bash
# VWAP leakage (CRITICAL!)
pytest tests/test_ml/test_data_leakage.py::TestFeatureLeakage::test_vwap_fixed_no_leakage -v

# Scaler leakage
pytest tests/test_ml/test_data_leakage.py::TestScalerLeakage::test_scaler_fit_only_on_train -v

# RSI leakage
pytest tests/test_ml/test_data_leakage.py::TestFeatureLeakage::test_rsi_uses_only_past_data -v

# Walk-forward
pytest tests/test_ml/test_data_leakage.py::TestWalkForwardValidation::test_sequential_validation_no_leakage -v
```

---

### Order Execution Tests (`tests/test_order_manager/`)

#### `test_async_executor.py` (18+ tests)
**Purpose:** Verify async order execution logic

**Test Classes:**
- `TestBasicOrderFlow` - Normal execution
- `TestOrderChasing` - Price update logic
- `TestTimeoutAndFallback` - Timeout handling
- `TestRaceConditions` - Race condition protection
- `TestPartialFills` - Partial fill handling
- `TestRetryLogic` - Network error retries
- `TestEdgeCases` - Edge cases

**Key Tests:**
```bash
# Basic flow
pytest tests/test_order_manager/test_async_executor.py::TestBasicOrderFlow::test_successful_maker_fill -v

# Race condition (CRITICAL!)
pytest tests/test_order_manager/test_async_executor.py::TestRaceConditions::test_order_fills_during_cancel_attempt -v

# Timeout handling
pytest tests/test_order_manager/test_async_executor.py::TestTimeoutAndFallback::test_timeout_converts_to_market -v

# Retry logic
pytest tests/test_order_manager/test_async_executor.py::TestRetryLogic::test_retry_on_network_error -v
```

---

## 🔬 Advanced Testing

### Run with Coverage
```bash
# Generate HTML coverage report
pytest tests/ --cov=src --cov-report=html

# Open in browser
open htmlcov/index.html
```

**Target Coverage:**
- Overall: > 80%
- Critical modules (labeling, executor): > 95%
- Utils/helpers: > 70%

---

### Run with Markers
```python
# In test file
@pytest.mark.slow
def test_large_dataset():
    # ...

@pytest.mark.critical
def test_data_leakage():
    # ...
```

```bash
# Run only critical tests
pytest tests/ -m critical

# Skip slow tests
pytest tests/ -m "not slow"
```

---

### Run with Debug Output
```bash
# Show print statements
pytest tests/ -s

# Stop on first failure
pytest tests/ -x

# Show full traceback
pytest tests/ --tb=long
```

---

## 🐛 Debugging Test Failures

### Common Issues

#### 1. Import Errors
```
ModuleNotFoundError: No module named 'pandas'
```

**Fix:**
```bash
pip install pandas numpy
```

---

#### 2. Assertion Failures
```
AssertionError: Expected label=1, got 0
```

**Debug:**
```python
# Add print statements
def test_take_profit_hit_first(self):
    labels = labeler.label(df)
    print(f"DEBUG: labels = {labels}")
    print(f"DEBUG: df = \n{df}")
    assert labels.iloc[0] == 1
```

Run with `-s` flag:
```bash
pytest tests/test_ml/test_triple_barrier.py::test_take_profit_hit_first -v -s
```

---

#### 3. Async Test Failures
```
RuntimeError: Event loop is closed
```

**Fix:** Ensure `pytest-asyncio` is installed:
```bash
pip install pytest-asyncio
```

**Verify async decorator:**
```python
@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_func()
    assert result is not None
```

---

## 📈 Test Metrics

### Current Coverage (After Fixes)

| Module | Coverage | Critical Tests |
|--------|----------|----------------|
| `labeling.py` | ~95% | ✅ 10+ tests |
| `feature_engineering.py` | ~90% | ✅ 8+ tests |
| `smart_limit_executor.py` | ~85% | ✅ 12+ tests |
| `order_executor.py` | ~80% | ✅ 5+ tests |

**Total Tests:** 53+
**Test LOC:** ~3,500 lines
**Test/Code Ratio:** ~0.6 (good)

---

## 🎯 Pre-Deployment Test Checklist

Before deploying to production, ensure:

### Critical Tests
- [ ] All data leakage tests pass
- [ ] All race condition tests pass
- [ ] All triple barrier tests pass
- [ ] All async executor tests pass

### Coverage
- [ ] Overall coverage > 80%
- [ ] Critical modules > 90%
- [ ] No untested critical paths

### Performance
- [ ] Test suite runs in < 60 seconds
- [ ] No flaky tests (run 10x, all pass)
- [ ] Memory usage acceptable

### Integration
- [ ] Tests run in CI/CD
- [ ] Tests run on fresh environment
- [ ] All dependencies documented

---

## 🔧 Writing New Tests

### Template for ML Tests
```python
# tests/test_ml/test_my_feature.py

import pytest
import pandas as pd
import numpy as np
from src.ml.my_module import MyClass

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'close': [100.0, 101.0, 99.0],
        'volume': [1000, 1000, 1000],
    })

def test_basic_functionality(sample_data):
    """Test basic case."""
    result = MyClass().process(sample_data)
    assert result is not None
    assert len(result) == len(sample_data)

def test_edge_case_empty_data():
    """Test edge case: empty data."""
    empty_df = pd.DataFrame()
    result = MyClass().process(empty_df)
    assert len(result) == 0
```

### Template for Async Tests
```python
# tests/test_order_manager/test_my_executor.py

import pytest
from unittest.mock import AsyncMock

@pytest.fixture
def mock_exchange():
    """Mock exchange API."""
    exchange = AsyncMock()
    exchange.create_order = AsyncMock(return_value={"id": "123"})
    return exchange

@pytest.mark.asyncio
async def test_place_order(mock_exchange):
    """Test order placement."""
    executor = MyExecutor()
    result = await executor.place_order(mock_exchange, order)

    assert result.success is True
    mock_exchange.create_order.assert_called_once()
```

---

## 📚 Resources

### Pytest Documentation
- https://docs.pytest.org/
- https://pytest-asyncio.readthedocs.io/

### Best Practices
- Keep tests independent (no shared state)
- Use fixtures for common setup
- Test one thing per test
- Use descriptive test names
- Mock external dependencies

### CI/CD Integration
```yaml
# .github/workflows/tests.yml

name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-asyncio pytest-cov
      - run: pytest tests/ --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v2
```

---

## ❓ FAQ

**Q: Tests are slow, how to speed up?**
A: Use `pytest-xdist` for parallel execution:
```bash
pip install pytest-xdist
pytest tests/ -n auto
```

**Q: How to test only changed files?**
A: Use `pytest-testmon`:
```bash
pip install pytest-testmon
pytest --testmon
```

**Q: How to generate test report?**
A: Use `pytest-html`:
```bash
pip install pytest-html
pytest tests/ --html=report.html
```

---

**Happy Testing! 🧪**
