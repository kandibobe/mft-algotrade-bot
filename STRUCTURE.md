# 📁 Stoic Citadel - Структура проекта

Детальное описание архитектуры, файлов и директорий HFT торгового бота.

---

## 🏗️ Общая структура

```
C:\hft-algotrade-bot\
├── 📂 docker/                    # Docker конфигурация
├── 📂 scripts/                   # Автоматизация
│   └── 📂 windows/               # PowerShell скрипты
├── 📂 user_data/                 # Данные пользователя
│   ├── 📂 config/                # Конфигурация
│   ├── 📂 data/                  # Исторические данные
│   ├── 📂 logs/                  # Логи
│   ├── 📂 strategies/            # Торговые стратегии
│   └── 📂 backtest_results/      # Результаты бэктестов
├── 📂 research/                  # Jupyter notebooks
├── 📄 docker-compose.yml         # Оркестрация сервисов
├── 📄 .env                       # Переменные окружения
├── 📄 QUICKSTART.md              # Быстрый старт
├── 📄 LOGS.md                    # Руководство по логам
├── 📄 STRUCTURE.md               # Этот файл
└── 📄 README.md                  # Главная документация
```

---

## 📂 Детальное описание директорий

### `/docker` - Docker конфигурация

```
docker/
└── Dockerfile.jupyter          # Сборка Jupyter Lab с quant библиотеками
```

**Dockerfile.jupyter**:
- **Базовый образ**: `jupyter/scipy-notebook:python-3.11`
- **TA-Lib**: Компиляция из исходников (v0.4.0)
- **Библиотеки**:
  - Freqtrade 2024.11
  - pandas-ta 0.3.14
  - scikit-learn, xgboost, lightgbm
  - polars, plotly, matplotlib
  - backtesting, optuna
- **Порт**: 8888
- **Token**: stoic2024

---

### `/scripts/windows` - PowerShell автоматизация

```
scripts/windows/
├── deploy.ps1           # Полное развертывание
├── backtest.ps1         # Запуск бэктестов
├── download-data.ps1    # Загрузка данных
└── logs.ps1             # Просмотр логов
```

#### **deploy.ps1**
**Назначение**: Автоматизированное развертывание всего проекта

**Параметры**:
- `-SkipData`: Пропустить загрузку данных
- `-SkipBacktest`: Пропустить бэктест
- `-WithJupyter`: Запустить Jupyter Lab
- `-AllServices`: Запустить все сервисы
- `-DataDays <int>`: Количество дней данных (по умолчанию 90)
- `-Strategy <string>`: Стратегия для бэктеста

**Примеры**:
```powershell
# Полное развертывание
.\scripts\windows\deploy.ps1

# Без загрузки данных
.\scripts\windows\deploy.ps1 -SkipData

# Со всеми сервисами
.\scripts\windows\deploy.ps1 -AllServices
```

#### **backtest.ps1**
**Назначение**: Гибкий запуск бэктестов

**Параметры**:
- `-Strategy <string>`: Имя стратегии
- `-Timerange <string>`: Период (YYYYMMDD-YYYYMMDD)
- `-StartDaysAgo <int>`: Дней назад от сегодня
- `-Pairs <string>`: Торговые пары
- `-MaxOpenTrades <int>`: Макс открытых позиций
- `-EnablePositionStacking`: Стекинг позиций
- `-ExportTrades`: Экспорт результатов
- `-Breakdown`: Детализация по дням/неделям

**Примеры**:
```powershell
# Базовый бэктест
.\scripts\windows\backtest.ps1 -Strategy "SimpleTestStrategy"

# С экспортом и детализацией
.\scripts\windows\backtest.ps1 -Strategy "StoicStrategyV1" -ExportTrades -Breakdown

# Кастомный период
.\scripts\windows\backtest.ps1 -Timerange "20241001-20241201"
```

#### **download-data.ps1**
**Назначение**: Загрузка исторических данных

**Параметры**:
- `-Days <int>`: Количество дней (по умолчанию 90)
- `-Timeframe <string>`: Таймфрейм (по умолчанию "5m")
- `-Exchange <string>`: Биржа (по умолчанию "binance")
- `-Pairs <string>`: Список пар
- `-WithBTC1d`: Загрузить BTC 1d для фильтра
- `-TradingViewFormat`: Формат TradingView JSON

**Примеры**:
```powershell
# Стандартная загрузка (90 дней, 5m)
.\scripts\windows\download-data.ps1

# Больше данных
.\scripts\windows\download-data.ps1 -Days 180

# Часовой таймфрейм
.\scripts\windows\download-data.ps1 -Timeframe "1h" -Days 365

# С BTC 1d для продвинутых стратегий
.\scripts\windows\download-data.ps1 -WithBTC1d
```

#### **logs.ps1**
**Назначение**: Просмотр и анализ логов

**Параметры**:
- `-Service <string>`: Сервис (freqtrade/frequi/jupyter/all)
- `-Lines <int>`: Количество строк (по умолчанию 50)
- `-Follow`: Следование за логами
- `-Timestamps`: Показать временные метки
- `-Level <string>`: Фильтр (ERROR/WARNING/INFO/DEBUG)
- `-Search <string>`: Поиск по тексту
- `-FileLog`: Просмотр файловых логов
- `-Export`: Экспорт в файл

**Примеры**:
```powershell
# Базовый просмотр
.\scripts\windows\logs.ps1

# Следование за логами
.\scripts\windows\logs.ps1 -Follow

# Только ошибки
.\scripts\windows\logs.ps1 -Level ERROR -Lines 200

# Файловые логи
.\scripts\windows\logs.ps1 -FileLog -Search "Strategy"
```

---

### `/user_data` - Данные пользователя

Главная директория с конфигурацией, данными и результатами.

#### `/user_data/config` - Конфигурация

```
user_data/config/
└── config.json          # Главный конфигурационный файл
```

**config.json** - основные параметры:

```json
{
  "dry_run": true,                    // Бумажная торговля
  "dry_run_wallet": 10000,            // Виртуальный баланс USDT
  "max_open_trades": 3,               // Макс позиций
  "stake_currency": "USDT",           // Валюта стейка
  "stake_amount": "unlimited",        // Размер позиции
  "tradable_balance_ratio": 0.99,     // % баланса для торговли
  "timeframe": "5m",                  // Таймфрейм свечей
  
  "exchange": {
    "name": "binance",                // Биржа
    "key": "",                        // API ключ (опционально)
    "secret": "",                     // API secret
    "ccxt_config": {},
    "ccxt_async_config": {}
  },
  
  "pair_whitelist": [                 // Торговые пары
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT"
  ],
  
  "stoploss": -0.05,                  // Стоплосс -5%
  
  "minimal_roi": {                    // ROI targets
    "0": 0.05,
    "150": 0.03,
    "300": 0.01
  },
  
  "api_server": {
    "enabled": true,
    "listen_ip_address": "0.0.0.0",
    "listen_port": 8080,
    "username": "stoic_admin",        // API логин
    "password": "StoicGuard2024"      // API пароль
  }
}
```

**Важные настройки для изменения**:
- `dry_run`: `false` для реальной торговли (⚠️ ОПАСНО!)
- `max_open_trades`: Количество позиций
- `pair_whitelist`: Список торгуемых пар
- `stoploss`: Уровень стоплосса

#### `/user_data/data` - Исторические данные

```
user_data/data/
└── binance/
    ├── BTC_USDT-5m.feather      # Парные данные 5m
    ├── BTC_USDT-1d.feather      # Парные данные 1d
    ├── ETH_USDT-5m.feather
    └── ...
```

**Формат**: Apache Feather (быстрый бинарный формат)  
**Размер**: ~0.5-1 MB на пару за 90 дней

**Управление**:
```powershell
# Посмотреть размер
Get-ChildItem -Path .\user_data\data\binance\ -Recurse | Measure-Object -Property Length -Sum

# Удалить старые данные
Remove-Item .\user_data\data\binance\* -Recurse

# Загрузить новые
.\scripts\windows\download-data.ps1
```

#### `/user_data/logs` - Логи

```
user_data/logs/
└── freqtrade.log        # Основной лог файл
```

**Ротация**: Автоматически управляется Freqtrade  
**Формат**: Plain text с временными метками  
**Размер**: Растет со временем, рекомендуется периодическая очистка

**Управление**:
```powershell
# Архивировать
Copy-Item .\user_data\logs\freqtrade.log ".\backups\logs\freqtrade_$(Get-Date -Format 'yyyyMMdd').log"

# Очистить
Clear-Content .\user_data\logs\freqtrade.log
```

#### `/user_data/strategies` - Торговые стратегии

```
user_data/strategies/
├── SimpleTestStrategy.py         # ⭐ Базовая (по умолчанию)
├── StoicStrategyV1.py            # 🚀 Продвинутая
├── StoicEnsembleStrategy.py      # 💎 Ансамбль
└── StoicCitadelV2.py             # ⚠️ В разработке
```

**SimpleTestStrategy.py**:
- **Индикаторы**: RSI
- **Логика**: Buy RSI<30, Sell RSI>70
- **Зависимости**: Нет
- **Статус**: ✅ Работает
- **Использование**: Тестирование инфраструктуры

**StoicStrategyV1.py**:
- **Индикаторы**: EMA, RSI, MACD, Bollinger Bands, ATR
- **Режимный фильтр**: BTC/USDT 1d EMA200
- **Логика**: Мульти-индикаторные сигналы
- **ROI**: Динамический на основе ATR
- **Зависимости**: Требует BTC/USDT 1d данные
- **HyperOpt**: Поддерживается
- **Статус**: ✅ Работает
- **Использование**: Продакшн-торговля

**StoicEnsembleStrategy.py**:
- **Тип**: Композиция из нескольких стратегий
- **Логика**: Голосование между стратегиями
- **Статус**: ⚠️ Экспериментальная

**StoicCitadelV2.py**:
- **Статус**: ❌ Требует исправления импортов
- **Проблема**: `No module named 'signals.indicators'`

**Создание своей стратегии**:
```powershell
# Копировать шаблон
Copy-Item .\user_data\strategies\SimpleTestStrategy.py .\user_data\strategies\MyStrategy.py

# Редактировать
code .\user_data\strategies\MyStrategy.py

# Изменить класс:
class MyStrategy(IStrategy):
    # Ваша логика
```

#### `/user_data/backtest_results` - Результаты бэктестов

```
user_data/backtest_results/
├── backtest-result-20241201-143022.json
└── backtest-result-20241202-091545.json
```

**Формат**: JSON с детальной статистикой  
**Содержит**:
- Общую прибыльность
- Статистику по парам
- Детали каждой сделки
- Метрики (Sharpe, Sortino, max drawdown)

---

### `/research` - Jupyter notebooks

```
research/
├── 01_strategy_template.ipynb    # Шаблон разработки стратегий
├── README.md                     # Руководство по исследованиям
└── (пользовательские notebooks)
```

**Использование**:
1. Запустить Jupyter: `docker-compose up -d jupyter`
2. Открыть: http://localhost:8888 (token: stoic2024)
3. Создать новый notebook для анализа

**Доступные данные в Jupyter**:
- `/home/jovyan/user_data` - данные бота (read-only)
- `/home/jovyan/strategies` - стратегии
- `/home/jovyan/research` - рабочая директория

---

## 🐳 Docker сервисы

### `freqtrade` - Торговый движок

**Image**: freqtradeorg/freqtrade:2024.11  
**Container**: stoic_freqtrade  
**Порты**: 8080 (API)  
**Volumes**:
- `./user_data:/freqtrade/user_data`

**Command**:
```bash
trade \
  --logfile /freqtrade/user_data/logs/freqtrade.log \
  --db-url sqlite:////freqtrade/user_data/tradesv3.sqlite \
  --config /freqtrade/user_data/config/config.json \
  --strategy SimpleTestStrategy
```

**Health Check**: `curl -f http://localhost:8080/api/v1/ping`

### `frequi` - Web Dashboard

**Image**: freqtradeorg/frequi:latest  
**Container**: stoic_frequi  
**Порты**: 3000 → 8080  
**Depends**: freqtrade (healthy)

**Доступ**: http://localhost:3000  
**Credentials**: stoic_admin / StoicGuard2024

### `jupyter` - Research Lab

**Image**: Custom (build from docker/Dockerfile.jupyter)  
**Container**: stoic_jupyter  
**Порты**: 8888  
**Volumes**:
- `./research:/home/jovyan/research`
- `./user_data:/home/jovyan/user_data:ro`
- `./scripts:/home/jovyan/scripts`
- `./user_data/strategies:/home/jovyan/strategies`

**Доступ**: http://localhost:8888  
**Token**: stoic2024

**Библиотеки**:
- freqtrade, pandas, numpy, polars
- TA-Lib, pandas-ta, technical
- scikit-learn, xgboost, lightgbm
- matplotlib, seaborn, plotly
- optuna, backtesting

### `postgres` - Analytics DB (Optional)

**Image**: postgres:16-alpine  
**Container**: stoic_postgres  
**Порты**: 5432  
**Credentials**: stoic_trader / StoicDB2024  
**Database**: trading_analytics

**Использование**: Для продвинутой аналитики и ML фич

### `portainer` - Container Management (Optional)

**Image**: portainer/portainer-ce:2.19.4  
**Container**: stoic_portainer  
**Порты**: 9443 (HTTPS), 9000 (HTTP)

**Доступ**: http://localhost:9443  
**Настройка**: При первом запуске

---

## 📄 Конфигурационные файлы

### `docker-compose.yml`

Главный файл оркестрации всех сервисов.

**Структура**:
```yaml
services:
  freqtrade:      # Торговый движок
  frequi:         # Web UI
  jupyter:        # Research
  postgres:       # Analytics DB
  portainer:      # Management

networks:
  stoic_network:  # Изолированная сеть

volumes:
  postgres_data:  # Данные PostgreSQL
  portainer_data: # Данные Portainer
```

### `.env`

Переменные окружения (опционально).

**Пример**:
```bash
# API Credentials
FREQTRADE_API_USERNAME=stoic_admin
FREQTRADE_API_PASSWORD=StoicGuard2024

# Jupyter
JUPYTER_TOKEN=stoic2024

# PostgreSQL
POSTGRES_PASSWORD=StoicDB2024

# Binance API (для реальной торговли)
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
```

---

## 🗄️ База данных

### `tradesv3.sqlite`

**Расположение**: `user_data/tradesv3.sqlite`  
**Тип**: SQLite3  
**Назначение**: Хранение истории сделок

**Таблицы**:
- `trades` - Открытые и закрытые сделки
- `orders` - История ордеров
- `pairlocks` - Блокировки пар

**Запросы**:
```powershell
# Установить SQLite (если нет)
winget install SQLite.SQLite

# Подключиться
sqlite3 .\user_data\tradesv3.sqlite

# Примеры запросов:
SELECT COUNT(*) FROM trades;
SELECT * FROM trades WHERE is_open=1;
SELECT pair, profit_ratio FROM trades ORDER BY profit_ratio DESC LIMIT 10;
```

---

## 📊 Workflow диаграмма

```
┌─────────────┐
│   Docker    │
│  Compose    │
└──────┬──────┘
       │
       ├─────────────────────────────────────────┐
       │                                         │
       ▼                                         ▼
┌─────────────┐                          ┌─────────────┐
│ Freqtrade   │◄────────────────────────►│   FreqUI    │
│   Engine    │         API              │  Dashboard  │
└──────┬──────┘                          └─────────────┘
       │
       │ reads/writes
       ▼
┌─────────────┐
│ user_data/  │
│ ├─config    │
│ ├─data      │
│ ├─logs      │
│ └─strategies│
└─────────────┘
       ▲
       │ analyzes
       │
┌──────┴──────┐
│   Jupyter   │
│     Lab     │
└─────────────┘
```

---

## 🔐 Безопасность

### Чувствительные данные

**НЕ коммитить в Git**:
- `.env` - переменные окружения
- `user_data/tradesv3.sqlite` - база сделок
- `user_data/logs/` - логи
- Файлы с API ключами

**.gitignore** должен содержать:
```
.env
user_data/tradesv3.sqlite
user_data/logs/
user_data/data/
user_data/backtest_results/
__pycache__/
*.pyc
```

### Хранение API ключей

**Безопасный способ (переменные окружения)**:
```powershell
# Установить временно
$env:BINANCE_API_KEY="your_key"
$env:BINANCE_API_SECRET="your_secret"

# Или в .env файл (НЕ коммитить!)
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
```

**В config.json**:
```json
{
  "exchange": {
    "name": "binance",
    "key": "${BINANCE_API_KEY}",
    "secret": "${BINANCE_API_SECRET}"
  }
}
```

---

## 📚 Дополнительные ресурсы

- **QUICKSTART.md**: Быстрый старт
- **LOGS.md**: Руководство по логам
- **README.md**: Главная документация
- **Freqtrade Docs**: https://www.freqtrade.io/en/stable/

---

**Успешной разработки! 🚀💻**
