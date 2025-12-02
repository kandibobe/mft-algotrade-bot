# 📁 Stoic Citadel - Структура проекта

## Общий обзор

```
hft-algotrade-bot/
├── 📁 docker/                      # Docker образы
├── 📁 scripts/                     # Автоматизация
├── 📁 user_data/                   # Данные пользователя (не коммитится)
├── 📁 research/                    # Jupyter ноутбуки для R&D
├── 📄 docker-compose.yml           # Оркестрация сервисов
├── 📄 .env                         # Переменные окружения
├── 📄 .gitignore                   # Игнорируемые файлы
├── 📄 README.md                    # Главная документация
├── 📄 QUICKSTART.md                # Быстрый старт (Windows)
├── 📄 STRUCTURE.md                 # Этот файл
└── 📄 LOGS.md                      # Гайд по логам
```

---

## 🐳 Docker конфигурация

### `/docker/`

Содержит Dockerfile'ы для кастомных образов.

```
docker/
└── Dockerfile.jupyter              # Jupyter Lab + TA-Lib + quant libs
```

**Dockerfile.jupyter**:
- **Базовый образ**: `jupyter/scipy-notebook:python-3.11`
- **Установлено**:
  - TA-Lib (компилируется из исходников)
  - Freqtrade 2024.11
  - Библиотеки ML: scikit-learn, xgboost, lightgbm
  - Библиотеки визуализации: plotly, matplotlib, seaborn
  - Backtesting.py, optuna, и другие quant tools
- **Порт**: 8888
- **Token**: stoic2024

---

## 🎛️ Оркестрация сервисов

### `/docker-compose.yml`

Определяет все сервисы проекта:

| Сервис | Образ | Порты | Статус | Описание |
|--------|-------|-------|--------|----------|
| **freqtrade** | freqtradeorg/freqtrade:2024.11 | 8080 | Required | Торговый движок |
| **frequi** | freqtradeorg/frequi:latest | 3000 | Required | Web dashboard |
| **jupyter** | custom build | 8888 | Optional | Research lab |
| **postgres** | postgres:16-alpine | 5432 | Optional | Analytics DB |
| **portainer** | portainer/portainer-ce:2.19.4 | 9443, 9000 | Optional | Container mgmt |

#### Зависимости между сервисами:

```
frequi ─depends_on→ freqtrade (healthcheck)
```

Все остальные сервисы независимы.

#### Volumes (постоянное хранилище):

- `./user_data` → `/freqtrade/user_data` (Freqtrade)
- `./research` → `/home/jovyan/research` (Jupyter)
- `postgres_data` → `/var/lib/postgresql/data` (PostgreSQL)
- `portainer_data` → `/data` (Portainer)

#### Networks:

- **stoic_network**: bridge network для всех сервисов

---

## 📊 Пользовательские данные

### `/user_data/`

**Главная директория** для всех торговых данных, стратегий, конфигов.

```
user_data/
├── config/
│   └── config.json                 # Главный конфиг Freqtrade
├── strategies/                     # Торговые стратегии (.py)
│   ├── SimpleTestStrategy.py       # Базовый RSI (по умолчанию)
│   ├── StoicStrategyV1.py          # Продвинутая стратегия
│   ├── StoicEnsembleStrategy.py    # Ансамбль стратегий
│   ├── StoicCitadelV2.py           # В разработке
│   └── __init__.py
├── data/
│   └── binance/                    # Исторические данные по парам
│       ├── BTC_USDT-5m.feather
│       ├── BTC_USDT-1d.feather
│       └── ...
├── logs/
│   └── freqtrade.log               # Основной лог файл
├── plot/                           # Графики (если используется plotting)
├── notebooks/                      # Пользовательские ноутбуки
└── tradesv3.sqlite                 # SQLite база сделок
```

#### `/user_data/config/config.json`

**Ключевые секции**:

```json
{
  "dry_run": true,                  // Режим симуляции
  "dry_run_wallet": 10000,          // Виртуальный баланс
  "max_open_trades": 3,             // Лимит открытых позиций
  "stake_currency": "USDT",
  "stake_amount": "unlimited",      // Автоматический sizing
  "tradable_balance_ratio": 0.99,   // Использовать 99% баланса
  "timeframe": "5m",
  "exchange": {
    "name": "binance",
    "key": "",                      // Пусто для dry_run
    "secret": ""
  },
  "pair_whitelist": [...],          // Список торговых пар
  "stoploss": -0.05,                // Глобальный стоплосс -5%
  "trailing_stop": false,
  "api_server": {                   // Настройки API (для FreqUI)
    "enabled": true,
    "listen_ip_address": "0.0.0.0",
    "listen_port": 8080,
    "username": "stoic_admin",
    "password": "StoicGuard2024"
  }
}
```

#### `/user_data/strategies/`

**Доступные стратегии**:

1. **SimpleTestStrategy.py** ⭐
   - RSI(14) oscillator
   - Buy: RSI < 30, Sell: RSI > 70
   - Timeframe: 5m
   - ROI: 5% immediate, 3% @150min, 1% @300min
   - Stoploss: -5%
   - **Статус**: Production-ready, по умолчанию

2. **StoicStrategyV1.py** 🚀
   - Market regime filter (BTC/USDT 1d EMA200)
   - Entry: RSI, MACD, ADX, volume
   - Exit: RSI extremes, MACD divergence
   - ATR-based position sizing
   - HyperOpt compatible
   - **Требует**: BTC/USDT 1d данные
   - **Статус**: Production-ready

3. **StoicEnsembleStrategy.py** 💎
   - Композиция из нескольких sub-strategies
   - Voting mechanism
   - **Статус**: Beta

4. **StoicCitadelV2.py** ⚠️
   - Advanced ML features
   - **Статус**: В разработке (import errors)

#### `/user_data/data/binance/`

Формат данных: **Feather** (Apache Arrow)

Пример файлов:
- `BTC_USDT-5m.feather` - 5-минутные свечи
- `BTC_USDT-1d.feather` - дневные свечи
- `ETH_USDT-5m.feather`
- и т.д.

**Загрузка**:
```powershell
docker-compose run --rm freqtrade download-data \
  --config /freqtrade/user_data/config/config.json \
  --exchange binance \
  --pairs BTC/USDT ETH/USDT \
  --timeframe 5m \
  --days 90
```

#### `/user_data/logs/freqtrade.log`

**Уровни логирования**:
- `INFO` - Обычные события
- `WARNING` - Предупреждения
- `ERROR` - Ошибки
- `CRITICAL` - Критичные сбои

**Ротация логов**: Автоматическая (ежедневно)

#### `/user_data/tradesv3.sqlite`

**SQLite база** со всеми сделками.

**Таблицы**:
- `trades` - История сделок
- `orders` - Ордера
- `pairlocks` - Блокировки пар

**Запросы**:
```sql
-- Все прибыльные сделки
SELECT * FROM trades WHERE close_profit_abs > 0;

-- Топ-10 пар по профиту
SELECT pair, SUM(close_profit_abs) as profit 
FROM trades 
GROUP BY pair 
ORDER BY profit DESC 
LIMIT 10;
```

---

## 🔬 Research & Development

### `/research/`

Jupyter ноутбуки для анализа и разработки стратегий.

```
research/
├── 01_strategy_template.ipynb      # Шаблон для новых стратегий
├── 02_data_exploration.ipynb       # (пример) Исследование данных
├── 03_backtest_analysis.ipynb      # (пример) Анализ результатов
└── README.md                       # Инструкции по R&D
```

**Доступ**: http://localhost:8888 (token: stoic2024)

**Пример использования**:

```python
import pandas as pd
from freqtrade.data.history import load_pair_history

# Загрузить исторические данные
df = load_pair_history(
    datadir='/home/jovyan/user_data/data',
    timeframe='5m',
    pair='BTC/USDT',
    exchange='binance'
)

# Анализировать
df['RSI'] = ta.RSI(df['close'], timeperiod=14)
df.plot(y=['close', 'RSI'], subplots=True)
```

---

## 🤖 Автоматизация

### `/scripts/windows/`

PowerShell скрипты для Windows.

```
scripts/windows/
├── deploy.ps1                      # Полное развертывание
├── backtest.ps1                    # Запуск бэктестов
├── download-data.ps1               # Загрузка данных
├── logs.ps1                        # Просмотр логов
└── README.md                       # Документация скриптов
```

#### `deploy.ps1`

**Полная автоматизация**:
1. Pull последних изменений
2. Build Jupyter (опционально)
3. Запуск Freqtrade + FreqUI
4. Health check
5. Загрузка тестовых данных
6. Запуск первого бэктеста

**Использование**:
```powershell
.\scripts\windows\deploy.ps1
```

#### `backtest.ps1`

**Параметры**:
- `-Strategy` - Имя стратегии (default: SimpleTestStrategy)
- `-Timerange` - Период (default: 20241001-)
- `-Config` - Путь к конфигу

**Использование**:
```powershell
.\scripts\windows\backtest.ps1 -Strategy "StoicStrategyV1" -Timerange "20241001-20241201"
```

#### `download-data.ps1`

**Параметры**:
- `-Days` - Количество дней (default: 90)
- `-Timeframe` - Таймфрейм (default: 5m)
- `-Pairs` - Список пар (default: BTC/USDT ETH/USDT ...)

**Использование**:
```powershell
.\scripts\windows\download-data.ps1 -Days 180 -Timeframe "1h"
```

#### `logs.ps1`

**Параметры**:
- `-Service` - Имя сервиса (default: freqtrade)
- `-Lines` - Количество строк (default: 100)
- `-Follow` - Следить в реальном времени

**Использование**:
```powershell
# Последние 100 строк
.\scripts\windows\logs.ps1 -Service "freqtrade"

# Следить в реальном времени
.\scripts\windows\logs.ps1 -Service "freqtrade" -Follow
```

---

## 🔐 Безопасность

### Файлы, НЕ попадающие в Git (`.gitignore`):

```
user_data/
!user_data/strategies/
!user_data/config/config.json
.env
*.sqlite
*.log
__pycache__/
.ipynb_checkpoints/
```

**Важно**:
- **API ключи** хранятся в `user_data/config/config.json` (не коммитится)
- **Переменные окружения** в `.env` (не коммитится)
- **Логи и данные** в `user_data/` (не коммитится)

### Рекомендации:

1. **Никогда не коммитить**:
   - API ключи и секреты
   - Файлы баз данных (*.sqlite)
   - Исторические данные (*.feather)
   - Логи (*.log)

2. **Использовать `.env` для secrets**:
   ```bash
   BINANCE_API_KEY=your_key_here
   BINANCE_API_SECRET=your_secret_here
   ```

3. **Dry run по умолчанию**:
   - Всегда начинать с `"dry_run": true`
   - Переключать на `false` только после тщательного тестирования

---

## 📈 Workflow разработки

### 1. Разработка новой стратегии

```
1. Jupyter Lab → Исследование данных
   ├── Загрузить исторические данные
   ├── Анализировать индикаторы
   └── Прототипировать логику

2. user_data/strategies/ → Создать .py файл
   ├── Скопировать SimpleTestStrategy.py
   ├── Реализовать populate_indicators()
   ├── Реализовать populate_entry_trend()
   └── Реализовать populate_exit_trend()

3. Backtesting → Тестировать
   ├── backtest.ps1 -Strategy "MyStrategy"
   ├── Анализировать метрики
   └── Итерировать

4. HyperOpt → Оптимизировать
   ├── docker-compose run --rm freqtrade hyperopt ...
   └── Применить лучшие параметры

5. Paper trading → Проверить на реальном рынке
   ├── dry_run: true
   ├── Мониторить 1-2 недели
   └── Анализировать расхождения с бэктестом

6. Production → Запуск с реальными деньгами
   └── dry_run: false (ОСТОРОЖНО!)
```

### 2. Обновление конфигурации

```
1. Редактировать user_data/config/config.json
2. Валидация: docker-compose config
3. Restart: docker-compose restart freqtrade
4. Проверка: docker-compose logs -f freqtrade
```

### 3. Добавление новых пар

```
1. Редактировать config.json → pair_whitelist
2. Загрузить данные: download-data.ps1
3. Бэктест с новыми парами
4. Restart: docker-compose restart freqtrade
```

---

## 🎓 Образовательные ресурсы

### Внутри проекта:

- `README.md` - Обзор проекта
- `QUICKSTART.md` - Быстрый старт для Windows
- `STRUCTURE.md` - Этот файл (детальная структура)
- `LOGS.md` - Гайд по логам и отладке
- `research/README.md` - Инструкции по R&D

### Внешние:

- [Freqtrade Docs](https://www.freqtrade.io/en/stable/)
- [TA-Lib Documentation](https://ta-lib.org/)
- [CCXT Exchange Support](https://github.com/ccxt/ccxt)

---

## 🔄 Обновления и поддержка

### Получение последних изменений:

```powershell
git pull origin simplify-architecture
docker-compose pull  # Обновить образы
docker-compose up -d --force-recreate
```

### Резервное копирование:

```powershell
# Бэкап пользовательских данных
Compress-Archive -Path .\user_data -DestinationPath backup_$(Get-Date -Format 'yyyyMMdd').zip

# Бэкап базы данных
copy .\user_data\tradesv3.sqlite .\backups\tradesv3_$(Get-Date -Format 'yyyyMMdd').sqlite
```

---

**Вопросы?** Создайте issue на GitHub: https://github.com/kandibobe/hft-algotrade-bot/issues
