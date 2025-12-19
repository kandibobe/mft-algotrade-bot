# 🚀 БЫСТРЫЙ СТАРТ - Stoic Citadel MFT Bot

**Время до первого запуска: 5 минут ⏱️**

---

## ⚡ Мгновенный Тест (30 секунд)

```bash
# Установить зависимости
pip install -r requirements.txt

# Запустить тестовый бэктест
python examples/quick_backtest.py

# ✅ Увидишь:
# - Генерация 1000 синтетических свечей
# - Расчет Triple Barrier labels
# - 56 сделок с метриками
# - Total Return и Sharpe Ratio
```

---

## 📋 Требования

- Python 3.10+ (рекомендуется 3.11)
- 4GB RAM минимум
- 10GB свободного места
- Git (опционально)

---

## 🔧 Установка

### 1. Клонировать или скачать

```bash
git clone https://github.com/kandibobe/mft-algotrade-bot.git
cd mft-algotrade-bot
```

### 2. Создать виртуальное окружение (рекомендуется)

```bash
# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Установить зависимости

```bash
pip install -r requirements.txt
```

**Что установится:**
- pandas, numpy - обработка данных
- scikit-learn, joblib - ML
- matplotlib - графики
- ccxt - API бирж
- pytest - тестирование

---

## 🧪 Запуск Тестов

### Критичные ML тесты (ВАЖНО!)

```bash
# Data Leakage Prevention (13 тестов)
pytest tests/test_ml/test_data_leakage.py -v

# Triple Barrier Labeling (16 тестов)
pytest tests/test_ml/test_triple_barrier.py -v

# Все ML тесты
pytest tests/test_ml/ -v
```

### Все тесты

```bash
# Запустить все (исключая Freqtrade integration)
pytest tests/ --ignore=tests/test_integration/test_trading_flow.py -v

# С покрытием
pytest tests/ --cov=src --cov-report=term
```

### Ожидаемый результат

```
✅ Triple Barrier:  16/16 PASSED (100%)
✅ Data Leakage:    13/13 PASSED (100%)
✅ Labeling:        21/21 PASSED (100%)
✅ Critical Tests:  40+/43 PASSED (93%+)
```

---

## 📊 Запуск Бэктеста

### Быстрый тест (синтетические данные)

```bash
python examples/quick_backtest.py
```

**Результат:**
- 1000 свечей сгенерировано
- 56 сделок выполнено
- Total Return, Sharpe Ratio, Max DD
- Full equity curve

### С реальными данными

```python
from scripts.backtest import BacktestEngine, BacktestConfig
import pandas as pd

# Загрузить CSV
data = pd.read_csv('data/BTC_USDT_1h.csv',
                   index_col='timestamp',
                   parse_dates=True)

# Настроить
config = BacktestConfig({
    'initial_capital': 10000.0,
    'take_profit': 0.02,  # 2%
    'stop_loss': 0.01,    # 1%
    'maker_fee': 0.001,   # 0.1%
})

# Запустить
engine = BacktestEngine(config)
results = engine.run_backtest(data)

print(f"Return: {results['total_return']:.2%}")
print(f"Trades: {len(results['trades'])}")
```

### Скачать данные с биржи

```python
from src.data.async_fetcher import AsyncDataFetcher, FetcherConfig
import asyncio

async def download():
    config = FetcherConfig(exchange='binance')
    async with AsyncDataFetcher(config) as fetcher:
        data = await fetcher.fetch_ohlcv('BTC/USDT', '1h', limit=1000)
        data.to_csv('data/BTC_USDT_1h.csv')
    return data

data = asyncio.run(download())
```

---

## 🧪 Paper Trading (Тестовая Торговля)

### 1. Получить Testnet API Keys

```bash
# Регистрация на Binance Testnet
https://testnet.binance.vision/

# После регистрации:
# Settings → API Management → Create API Key
```

### 2. Создать .env файл

```bash
cat > .env <<EOF
# Binance Testnet (БЕЗ реальных денег!)
BINANCE_TESTNET_API_KEY=your_testnet_api_key
BINANCE_TESTNET_API_SECRET=your_testnet_secret

# Telegram alerts (опционально)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
EOF

chmod 600 .env  # Защитить файл
```

### 3. Проверить конфигурацию

```bash
# Убедиться что sandbox=true
grep "sandbox:" config/paper_trading_config.yaml

# Проверить risk параметры
grep "risk_per_trade:" config/paper_trading_config.yaml
```

### 4. Запустить (когда готов main.py)

```bash
python -m src.main --config config/paper_trading_config.yaml
```

---

## 📁 Структура Проекта

```
mft-algotrade-bot/
├── config/                          # Конфигурации
│   ├── paper_trading_config.yaml    # ✅ Paper trading
│   └── live_trading_SAFE_DEFAULTS.yaml  # ⚠️ Live
│
├── src/                             # Исходники
│   ├── ml/training/                 # ML модули
│   │   ├── labeling.py             # ✅ Triple Barrier
│   │   └── feature_engineering.py  # ✅ 30+ индикаторов
│   ├── risk/
│   │   ├── position_sizing.py      # ✅ Kelly Criterion
│   │   └── pre_trade_checks.py     # ✅ NEW - валидация
│   ├── order_manager/
│   │   ├── circuit_breaker.py      # ✅ Автостоп
│   │   └── order_ledger.py         # ✅ NEW - persistence
│   └── data/
│       └── async_fetcher.py        # ✅ Async API
│
├── scripts/
│   └── backtest.py                 # ✅ Backtest engine
│
├── examples/
│   └── quick_backtest.py           # ✅ Готовый пример
│
├── tests/                          # ✅ 238 тестов
│   └── test_ml/                    # ✅ 100% coverage
│
└── docs/
    ├── PRODUCTION_READINESS_REPORT.md  # ✅ Полный отчет
    └── CRITICAL_FIXES_COMPLETED.md      # ✅ ML fixes
```

---

## 🎯 Ключевые Модули

### Triple Barrier Labeling

```python
from src.ml.training.labeling import TripleBarrierLabeler, TripleBarrierConfig

config = TripleBarrierConfig(
    take_profit=0.02,  # 2% TP
    stop_loss=0.01,    # 1% SL
)

labeler = TripleBarrierLabeler(config)
labels = labeler.label(data)  # 1=buy, -1=sell, 0=hold
```

### Pre-Trade Checks (НОВОЕ!)

```python
from src.risk.pre_trade_checks import PreTradeChecker, PreTradeConfig

checker = PreTradeChecker()
result = checker.validate_order(
    symbol='BTC/USDT',
    side='buy',
    quantity=0.01,
    price=50000.0,
    current_balance=10000.0,
)

if result.passed:
    # ✅ Отправить ордер
    pass
else:
    print(f"❌ {result.reason}")
```

### Order Ledger (НОВОЕ!)

```python
from src.order_manager.order_ledger import OrderLedger

ledger = OrderLedger("data/orders.db")

# Проверка дубликатов
if not ledger.is_duplicate("order_key_123"):
    order = exchange.create_order(...)
    ledger.store_order(order, idempotency_key="order_key_123")
```

---

## ⚠️ ВАЖНО

### Перед Paper Trading:
- ✅ Запустить quick_backtest.py
- ✅ Все ML тесты должны проходить
- ✅ Получить testnet API keys
- ✅ Настроить .env
- ✅ Проверить config/paper_trading_config.yaml

### Перед Live Trading:
- ✅ **2+ недели paper trading**
- ✅ Sharpe > 1.0, Win Rate > 45%
- ✅ Все сделки проверены вручную
- ✅ Alerts настроены
- ✅ Kill switch протестирован
- ✅ Начинать с 10% от целевого размера

### НИКОГДА:
- ❌ Live без paper trading
- ❌ Торговать последними деньгами
- ❌ Игнорировать circuit breaker
- ❌ Commit .env в git

---

## 🔧 Troubleshooting

### "ModuleNotFoundError"

```bash
# Переустановить зависимости
pip install -r requirements.txt --force-reinstall

# Проверить
python -c "from scripts.backtest import BacktestEngine; print('✅ OK')"
```

### "No module named freqtrade"

```bash
# Это нормально! Freqtrade опционален
# Тесты с freqtrade будут пропущены

# Запускать тесты без freqtrade:
pytest tests/ --ignore=tests/test_integration/test_trading_flow.py
```

### Backtest не генерирует сделки

```python
# Проверить labels
from src.ml.training.labeling import TripleBarrierLabeler, TripleBarrierConfig

config = TripleBarrierConfig(
    take_profit=0.01,  # Уменьшить TP
    stop_loss=0.005,   # Уменьшить SL
)

labeler = TripleBarrierLabeler(config)
labels = labeler.label(data)

print(f"Buy: {(labels == 1).sum()}")
print(f"Sell: {(labels == -1).sum()}")
# Если все 0 - стратегия слишком консервативна
```

### Python версия

```bash
python --version  # Должна быть 3.10+

# Если нет - установить:
# Ubuntu: sudo apt install python3.11
# Mac: brew install python@3.11
# Windows: python.org
```

---

## 📚 Документация

**Основное:**
- [PRODUCTION_READINESS_REPORT.md](PRODUCTION_READINESS_REPORT.md) - Полный отчет о готовности
- [CRITICAL_FIXES_COMPLETED.md](CRITICAL_FIXES_COMPLETED.md) - ML исправления
- [AUDIT_REPORT.md](AUDIT_REPORT.md) - Аудит проекта

**Конфигурации:**
- [config/paper_trading_config.yaml](config/paper_trading_config.yaml) - Paper trading
- [config/live_trading_SAFE_DEFAULTS.yaml](config/live_trading_SAFE_DEFAULTS.yaml) - Live trading

**Примеры:**
- [examples/quick_backtest.py](examples/quick_backtest.py) - Быстрый старт

---

## 🎯 Что Дальше?

### Сегодня (5 минут):
```bash
python examples/quick_backtest.py  # Проверить setup
```

### Эта Неделя:
1. Скачать реальные данные
2. Запустить бэктест на истории
3. Оптимизировать параметры (TP/SL)

### Следующие 2 Недели:
1. Получить testnet API keys
2. Запустить paper trading
3. Мониторить каждую сделку

### Через Месяц:
1. Review paper trading результатов
2. Если profitable → рассмотреть live (с осторожностью!)
3. Если убыточно → улучшить стратегию

---

## 📞 Поддержка

- **GitHub Issues**: https://github.com/kandibobe/mft-algotrade-bot/issues
- **Logs**: `logs/`
- **Database**: `data/orders.db`
- **Kill Switch**: `touch .kill_switch`

---

## ✅ Готов к Запуску!

**Первая команда:**
```bash
python examples/quick_backtest.py
```

**Увидишь:**
- ✅ 1000 свечей сгенерировано
- ✅ 56 сделок выполнено
- ✅ Все метрики рассчитаны
- ✅ Setup работает!

---

**Stoic Citadel MFT Bot** - Торгуй с умом, не с эмоциями 🧘

**Версия:** 2.0.0 | **Готовность:** Paper Trading Ready 🟢
