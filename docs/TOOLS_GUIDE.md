# 🛠️ Tools & Utilities Guide

Comprehensive guide to all available tools and utilities.

---

## 📊 Data Tools

### 1. Data Inspector

**Purpose:** Просмотр и анализ скачанных исторических данных

**Location:** `scripts/inspect_data.py`

**Usage:**

```bash
# Показать все доступные данные
python scripts/inspect_data.py

# Инспектировать конкретную пару
python scripts/inspect_data.py --pair BTC/USDT --timeframe 5m

# Сравнить несколько пар
python scripts/inspect_data.py --compare BTC/USDT ETH/USDT SOL/USDT
```

**Output Example:**
```
📊 DATA INSPECTION: BTC/USDT (5m)
======================================================================

📅 Time Range:
   Start:    2025-11-19 00:00:00
   End:      2025-12-19 14:33:21
   Duration: 30 days
   Candles:  8,815

💰 Price Statistics:
   Current:  $106,823.45
   High:     $108,245.12
   Low:      $91,234.56
   Avg:      $99,456.78
   Std Dev:  $4,234.56

📈 Returns:
   Total:    +15.23%
   Daily Avg:+0.51%
   Volatility:2.34%
   Max Gain: +8.45%
   Max Loss: -6.78%
```

---

### 2. Quick Backtest Runner

**Purpose:** Быстрый запуск бэктестов с готовыми профилями

**Location:** `scripts/run_backtest.py`

**Usage:**

```bash
# Показать все профили
python scripts/run_backtest.py --list-profiles

# Быстрый тест (7 дней, BTC/USDT)
python scripts/run_backtest.py --profile quick

# Полный тест (30 дней, BTC + ETH)
python scripts/run_backtest.py --profile full

# Aggressive test (volatile coins)
python scripts/run_backtest.py --profile aggressive

# Custom backtest
python scripts/run_backtest.py --pair BTC/USDT ETH/USDT --days 14 --timeframe 5m
```

**Available Profiles:**
- `quick` - 7 дней, BTC/USDT only
- `full` - 30 дней, BTC/USDT + ETH/USDT
- `aggressive` - 14 дней, volatile coins (SOL, AVAX, NEAR)
- `stable` - 30 дней, крупные монеты, 15m timeframe
- `all` - 30 дней, все основные пары

---

### 3. Quick Backtest (Synthetic Data)

**Purpose:** Тестирование setup с синтетическими данными

**Location:** `examples/quick_backtest.py`

**Usage:**

```bash
python examples/quick_backtest.py
```

**Output:**
- Генерирует 1000 свечей синтетических данных
- Запускает Triple Barrier labeling
- Обучает ML модель (если есть данные)
- Показывает результаты бэктеста
- Сохраняет отчет в `reports/quick_backtest.png`

---

## 🔧 Setup & Configuration

### Download Data

```bash
# Single pair, 30 days
docker exec stoic_freqtrade freqtrade download-data \
  --exchange binance \
  --timeframe 5m \
  --pairs BTC/USDT \
  --days 30

# Multiple pairs
docker exec stoic_freqtrade freqtrade download-data \
  --exchange binance \
  --timeframe 5m \
  --pairs BTC/USDT ETH/USDT SOL/USDT AVAX/USDT \
  --days 30

# Multiple timeframes
docker exec stoic_freqtrade freqtrade download-data \
  --exchange binance \
  --timeframe 5m 15m 1h \
  --pairs BTC/USDT ETH/USDT \
  --days 30
```

### List Downloaded Data

```bash
# Via Docker
docker exec stoic_freqtrade freqtrade list-data

# Via script
python scripts/inspect_data.py
```

---

## 📈 Running Backtests

### Method 1: Quick Runner (Recommended)

```bash
# Using profile
python scripts/run_backtest.py --profile quick

# Custom parameters
python scripts/run_backtest.py \
  --pair BTC/USDT ETH/USDT \
  --days 14 \
  --timeframe 5m \
  --strategy StoicEnsembleStrategyV2
```

### Method 2: Direct Freqtrade

```bash
docker exec stoic_freqtrade freqtrade backtesting \
  --strategy StoicEnsembleStrategyV2 \
  --timeframe 5m \
  --timerange 20251119-20251219 \
  --export trades
```

### Method 3: Python Script

```bash
python examples/quick_backtest.py
```

---

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Module

```bash
# Order manager tests
pytest tests/test_order_manager/ -v

# ML tests
pytest tests/test_ml/ -v

# Circuit breaker tests
pytest tests/test_risk/test_circuit_breaker.py -v
```

### Run Tests in Parallel

```bash
# 4 workers
pytest tests/ -v -n 4

# Auto-detect CPU count
pytest tests/ -v -n auto
```

### Skip Slow Tests

```bash
pytest tests/ -v -m "not slow"
```

---

## 🔍 Monitoring & Analysis

### Check Bot Status

```bash
# Container status
docker ps

# Bot logs
docker logs stoic_freqtrade --tail 100 -f

# FreqUI logs
docker logs stoic_frequi --tail 50
```

### Database Inspection

```bash
# Connect to PostgreSQL
docker exec -it stoic_postgres psql -U stoic_trader -d trading_analytics

# List tables
\dt

# Query trades
SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10;
```

### Redis Inspection

```bash
# Connect to Redis
docker exec -it stoic_redis redis-cli

# List keys
KEYS *

# Get value
GET ml_prediction:BTC_USDT

# Monitor commands
MONITOR
```

---

## 📊 Performance Analysis

### View Backtest Results

1. **Via FreqUI:**
   - Open http://localhost:3000
   - Login: stoic_admin / StoicGuard2024!ChangeMe
   - Go to "Backtesting" tab
   - Load latest results

2. **Via Files:**
   ```bash
   # List results
   ls -lh user_data/backtest_results/

   # View JSON
   cat user_data/backtest_results/backtest-result-*.json | jq .
   ```

3. **Via Python:**
   ```python
   import pandas as pd

   results = pd.read_json('user_data/backtest_results/backtest-result-latest.json')
   print(results.describe())
   ```

---

## 🚨 Troubleshooting

### Data Not Found

```bash
# Check if data directory exists
ls -la user_data/data/binance/

# Download data
python scripts/run_backtest.py --profile quick
# Will show download command if data missing
```

### Backtest Fails

```bash
# Check strategy is valid
docker exec stoic_freqtrade freqtrade list-strategies

# Validate config
docker exec stoic_freqtrade freqtrade show-config

# Check data quality
python scripts/inspect_data.py --pair BTC/USDT
```

### Tests Hanging

```bash
# Kill hanging tests
Ctrl+C

# Run specific test
pytest tests/test_order_manager/test_order_types.py::TestOrder::test_order_creation -v

# Skip problematic tests
pytest tests/ -v --ignore=tests/test_order_manager/test_async_executor.py
```

---

## 💡 Tips & Best Practices

### 1. Data Management

- Download data in batches (don't download 365 days at once)
- Use 5m timeframe for main testing (good balance)
- Keep at least 30 days of data for meaningful backtests
- Re-download data periodically (market conditions change)

### 2. Backtesting

- Start with `--profile quick` (7 days) for fast iteration
- Use `--profile full` (30 days) for realistic results
- Always export trades: `--export trades`
- Compare multiple strategies on same data

### 3. Development

- Use `quick_backtest.py` for rapid testing during development
- Run tests before committing: `pytest tests/ -v`
- Use `inspect_data.py` to understand your data
- Check bot logs regularly: `docker logs stoic_freqtrade -f`

### 4. Production

- Start with paper trading (dry_run = true)
- Monitor for at least 2 weeks before going live
- Set conservative position sizes initially
- Use circuit breaker (enabled by default)
- Enable Telegram notifications

---

## 📚 Additional Resources

- [QUICKSTART.md](../QUICKSTART.md) - Getting started guide
- [API_SETUP_RU.md](API_SETUP_RU.md) - API setup (Russian)
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Comprehensive testing guide
- [Freqtrade Docs](https://www.freqtrade.io/en/stable/) - Official docs

---

**Last Updated:** 2025-12-19
**Version:** 2.0 - Production Ready
