# Scripts Guide
## Stoic Citadel - Utility Scripts

Все скрипты для управления торговым ботом.

---

## 🚀 Quick Start Scripts

### **overnight_setup.ps1** (⭐ РЕКОМЕНДУЕТСЯ)
**Полная автоматическая настройка системы на ночь**

```powershell
# Скачать данные, обучить модели, запустить бэктесты
.\scripts\overnight_setup.ps1

# Только для конкретных пар
.\scripts\overnight_setup.ps1 -Pairs "BTC/USDT","ETH/USDT" -Days 30

# С оптимизацией гиперпараметров (займет больше времени)
.\scripts\overnight_setup.ps1 -Optimize
```

**Что делает:**
1. ✅ Скачивает 30 дней исторических данных
2. ✅ Обучает ML модели для каждой пары
3. ✅ Запускает бэктесты
4. ✅ Генерирует отчеты

**Время выполнения:** 30-120 минут (зависит от количества пар и оптимизации)

---

## 📥 Data Management

### **download_fresh_data.ps1**
**Скачивает СВЕЖИЕ данные (удаляет старые)**

```powershell
# Скачать 30 дней BTC и ETH
.\scripts\download_fresh_data.ps1 -Pairs "BTC/USDT ETH/USDT" -Days 30
```

**Особенности:**
- Удаляет старые данные в Docker
- Скачивает полные 30 дней (не обновление)
- Автоматически синхронизирует с локальным FS
- Показывает количество свечей

### **download_data.py**
**Python wrapper для скачивания с пресетами**

```bash
# Preset для major coins (BTC, ETH, BNB)
python scripts/download_data.py --preset major --days 30

# Preset для DeFi токенов
python scripts/download_data.py --preset defi --days 30

# Все популярные пары
python scripts/download_data.py --preset all --days 30
```

**Presets:**
- `major` - BTC, ETH, BNB
- `layer1` - SOL, AVAX, NEAR, ADA
- `defi` - UNI, LINK, AAVE, CRV
- `meme` - DOGE, SHIB, PEPE
- `all` - Все 12 популярных пар

### **sync_data.ps1**
**Синхронизирует данные из Docker в локальную файловую систему**

```powershell
.\scripts\sync_data.ps1
```

### **inspect_data.py**
**Просмотр статистики скачанных данных**

```bash
# Показать все доступные данные
python scripts/inspect_data.py

# Инспектировать конкретную пару
python scripts/inspect_data.py --pair BTC/USDT --timeframe 5m

# Сравнить несколько пар
python scripts/inspect_data.py --compare BTC/USDT ETH/USDT BNB/USDT
```

**Показывает:**
- 📅 Временной диапазон данных
- 💰 Статистику цен (high, low, avg, volatility)
- 📈 Returns (total, daily avg, max gain/loss)
- 📊 Volume statistics
- ✅ Data quality (missing values, gaps)

### **verify_data.py**
**Проверяет качество данных (gaps, anomalies, spikes)**

```bash
python scripts/verify_data.py
```

---

## 🤖 Machine Learning

### **train_models.py** (⭐ ВАЖНО)
**Обучает ML модели на исторических данных**

```bash
# Обучить модели для BTC и ETH
python scripts/train_models.py --pairs BTC/USDT ETH/USDT

# Quick mode (только последние 1000 свечей)
python scripts/train_models.py --pairs BTC/USDT --quick

# С оптимизацией гиперпараметров
python scripts/train_models.py --pairs BTC/USDT --optimize --trials 50
```

**Процесс:**
1. Загружает данные из JSON
2. Feature engineering (100+ features)
3. Triple Barrier labeling (LONG/NEUTRAL/SHORT)
4. Обучение Random Forest модели
5. Сохранение модели в `user_data/models/`

**Время выполнения:**
- Quick mode: ~2-5 минут на пару
- Full mode: ~10-20 минут на пару
- С оптимизацией: ~30-60 минут на пару

---

## 📊 Backtesting

### **run_backtest.py** (⭐ ПРОСТОЙ ЗАПУСК)
**Быстрый запуск бэктестов с профилями**

```bash
# Quick test (7 дней, BTC)
python scripts/run_backtest.py --profile quick

# Full test (30 дней, BTC + ETH)
python scripts/run_backtest.py --profile full

# Aggressive (14 дней, volatile coins)
python scripts/run_backtest.py --profile aggressive

# Кастомный тест
python scripts/run_backtest.py --pair BTC/USDT ETH/USDT --days 14 --timeframe 5m
```

**Профили:**
- `quick` - BTC, 7 дней, 5m (быстрый тест)
- `full` - BTC+ETH, 30 дней, 5m (полный тест)
- `aggressive` - SOL+AVAX+NEAR, 14 дней, 5m (высокая волатильность)
- `stable` - BTC+ETH+BNB, 30 дней, 15m (стабильные монеты)
- `all` - Все 7 пар, 30 дней, 5m (комплексный тест)

### **backtest.py**
**Production-ready backtesting с walk-forward validation**

```bash
python scripts/backtest.py --config config/backtest_config.json
python scripts/backtest.py --symbol BTC/USDT --start 2024-01-01 --end 2024-12-31
```

**Features:**
- Walk-forward validation
- Transaction costs
- Slippage simulation
- Performance metrics (Sharpe, Sortino, Max DD)
- Visual reports

---

## 🧪 Testing & Validation

### **smoke_test.py**
**Быстрый тест всей системы**

```bash
python scripts/smoke_test.py
```

**Проверяет:**
- Docker containers
- Database connections
- Exchange API
- Data availability
- Configuration files

### **run_tests.ps1**
**Запуск unit tests**

```powershell
.\scripts\run_tests.ps1
```

### **health_check.py**
**Проверка здоровья системы**

```bash
python scripts/health_check.py
```

---

## 🔧 Configuration & Setup

### **setup_wizard.py**
**Интерактивный wizard для первоначальной настройки**

```bash
python scripts/setup_wizard.py
```

### **validate_config.py**
**Валидация конфигурационных файлов**

```bash
python scripts/validate_config.py
```

---

## 📈 Optimization

### **optimize_strategy.py**
**Оптимизация параметров стратегии**

```bash
python scripts/optimize_strategy.py --strategy StoicEnsembleStrategyV2
```

### **walk_forward_validation.py**
**Walk-forward оптимизация и валидация**

```bash
python scripts/walk_forward_validation.py
```

---

## 🛠️ Utilities

### **generate_report.py**
**Генерация отчетов по бэктестам**

```bash
python scripts/generate_report.py --backtest-results user_data/backtest_results/
```

### **health_monitor.py**
**Continuous health monitoring**

```bash
python scripts/health_monitor.py --interval 60
```

---

## 📋 PowerShell Scripts

### **stoic.ps1**
**Главный CLI для управления системой**

```powershell
.\scripts\stoic.ps1 status
.\scripts\stoic.ps1 start
.\scripts\stoic.ps1 stop
.\scripts\stoic.ps1 logs
```

### **quick-start.ps1**
**Быстрый старт для новых пользователей**

```powershell
.\scripts\quick-start.ps1
```

### **health.ps1**
**Health check через PowerShell**

```powershell
.\scripts\health.ps1
```

---

## 🎯 Recommended Workflow

### Первый запуск:
```powershell
# 1. Запустить overnight setup
.\scripts\overnight_setup.ps1 -Pairs "BTC/USDT","ETH/USDT"

# 2. Утром проверить результаты
python scripts/inspect_data.py --compare BTC/USDT ETH/USDT

# 3. Открыть FreqUI
http://localhost:3000
```

### Ежедневное обновление:
```powershell
# Обновить данные (только новые свечи)
docker exec stoic_freqtrade freqtrade download-data --exchange binance --timeframe 5m --pairs BTC/USDT --days 1

# Переобучить модели раз в неделю
python scripts/train_models.py --pairs BTC/USDT ETH/USDT
```

### Тестирование новой стратегии:
```bash
# 1. Скачать свежие данные
.\scripts\download_fresh_data.ps1 -Days 30

# 2. Обучить модели
python scripts/train_models.py --pairs BTC/USDT --optimize

# 3. Запустить backtest
python scripts/run_backtest.py --profile full

# 4. Проверить результаты в FreqUI
```

---

## ⚙️ Environment Setup

Все скрипты требуют:
1. **Python 3.11+** с virtual environment
2. **Docker Desktop** (для freqtrade)
3. **PowerShell** (для .ps1 скриптов)

```powershell
# Activate venv before Python scripts
.\.venv\Scripts\Activate.ps1

# Check Docker
docker ps
```

---

## 📚 See Also

- [QUICKSTART_WINDOWS.md](../QUICKSTART_WINDOWS.md) - Windows setup guide
- [TOOLS_GUIDE.md](../docs/TOOLS_GUIDE.md) - Detailed tools documentation
- [CREDENTIALS.md](../CREDENTIALS.md) - Access credentials

---

**Last Updated:** 2025-12-19
