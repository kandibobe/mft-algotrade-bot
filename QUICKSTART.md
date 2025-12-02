# 🚀 Stoic Citadel - Быстрый старт (Windows)

## 📋 Требования

- **Docker Desktop** 4.25+ (с WSL2 backend)
- **PowerShell** 5.1+ (встроен в Windows)
- **Git** 2.40+
- **RAM**: минимум 8GB (рекомендуется 16GB)
- **Диск**: минимум 10GB свободного места

## 🎯 Быстрый запуск за 3 команды

```powershell
# 1. Клонировать репозиторий (если еще не сделано)
git clone https://github.com/kandibobe/hft-algotrade-bot.git
cd hft-algotrade-bot
git checkout simplify-architecture

# 2. Запустить основные сервисы
docker-compose up -d freqtrade frequi

# 3. Проверить статус (подождать 30 секунд)
docker-compose ps
```

### ✅ Доступ к интерфейсам

| Сервис | URL | Логин | Пароль |
|--------|-----|-------|--------|
| **FreqUI Dashboard** | http://localhost:3000 | stoic_admin | StoicGuard2024 |
| **Jupyter Lab** | http://localhost:8888 | - | Token: stoic2024 |
| **API** | http://localhost:8080/api/v1/ping | stoic_admin | StoicGuard2024 |
| **Portainer** | http://localhost:9443 | - | Setup on first run |

---

## 📊 Тестирование стратегий

### Вариант 1: Быстрый тест (без загрузки данных)

```powershell
# Протестировать SimpleTestStrategy на доступных данных
docker-compose run --rm freqtrade backtesting `
  --config /freqtrade/user_data/config/config.json `
  --strategy SimpleTestStrategy `
  --timerange 20241101-
```

### Вариант 2: Полноценный бэктест с загрузкой данных

```powershell
# Шаг 1: Загрузить 90 дней исторических данных (5m)
docker-compose run --rm freqtrade download-data `
  --config /freqtrade/user_data/config/config.json `
  --exchange binance `
  --pairs BTC/USDT ETH/USDT BNB/USDT SOL/USDT XRP/USDT `
  --timeframe 5m `
  --days 90

# Шаг 2: Запустить бэктест за последние 2 месяца
docker-compose run --rm freqtrade backtesting `
  --config /freqtrade/user_data/config/config.json `
  --strategy SimpleTestStrategy `
  --timerange 20241001-20241202
```

### Вариант 3: Продвинутая стратегия (требует BTC 1d данные)

```powershell
# Загрузить годовые дневные данные BTC для режимного фильтра
docker-compose run --rm freqtrade download-data `
  --config /freqtrade/user_data/config/config.json `
  --exchange binance `
  --pairs BTC/USDT `
  --timeframe 1d `
  --days 365

# Загрузить 5m данные для торговых пар
docker-compose run --rm freqtrade download-data `
  --config /freqtrade/user_data/config/config.json `
  --exchange binance `
  --pairs BTC/USDT ETH/USDT BNB/USDT SOL/USDT XRP/USDT `
  --timeframe 5m `
  --days 90

# Запустить бэктест с продвинутой стратегией
docker-compose run --rm freqtrade backtesting `
  --config /freqtrade/user_data/config/config.json `
  --strategy StoicStrategyV1 `
  --timerange 20241001-
```

---

## 🎛️ Управление сервисами

```powershell
# Запустить все сервисы (кроме optional)
docker-compose up -d freqtrade frequi

# Запустить с Jupyter Lab
docker-compose up -d freqtrade frequi jupyter

# Запустить ВСЕ сервисы (включая PostgreSQL, Portainer)
docker-compose up -d

# Остановить все
docker-compose down

# Перезапустить Freqtrade
docker-compose restart freqtrade

# Посмотреть логи
docker-compose logs -f freqtrade
docker-compose logs -f --tail=100 freqtrade

# Посмотреть статус
docker-compose ps
```

---

## 🔄 Смена стратегии

### Доступные стратегии

Расположение: `user_data/strategies/`

1. **SimpleTestStrategy.py** ⭐ (по умолчанию)
   - Базовый RSI-осциллятор
   - Не требует дополнительных данных
   - Идеален для тестирования инфраструктуры

2. **StoicStrategyV1.py** 🚀 (продвинутая)
   - Режимный фильтр на основе BTC/USDT 1d EMA200
   - Совместима с HyperOpt
   - Мульти-индикаторные входы/выходы
   - Кастомный sizing на базе ATR
   - **Требует**: BTC/USDT данные на 1d таймфрейме

3. **StoicEnsembleStrategy.py** 💎 (ансамбль)
   - Композиция из нескольких стратегий
   - Продвинутые фичи

4. **StoicCitadelV2.py** ⚠️ (в разработке)
   - Требует исправления импортов

### Как переключить стратегию

**Способ 1: Редактировать docker-compose.yml**

```yaml
# Найти строку в docker-compose.yml:
command: >
  trade
  --strategy SimpleTestStrategy  # <- Изменить здесь

# Изменить на:
  --strategy StoicStrategyV1

# Перезапустить:
docker-compose down
docker-compose up -d freqtrade frequi
```

**Способ 2: Через переменную окружения**

```powershell
$env:STRATEGY="StoicStrategyV1"
docker-compose up -d freqtrade frequi
```

---

## 🛠️ Автоматизация через PowerShell скрипты

### Использование helper-скриптов

```powershell
# Полное развертывание (deploy + download data + backtest)
.\scripts\windows\deploy.ps1

# Загрузить данные для определенного периода
.\scripts\windows\download-data.ps1 -Days 90 -Timeframe "5m"

# Запустить бэктест
.\scripts\windows\backtest.ps1 -Strategy "SimpleTestStrategy" -Timerange "20241001-"

# Просмотреть логи
.\scripts\windows\logs.ps1 -Service "freqtrade" -Lines 100
```

---

## 📁 Структура проекта

```
C:\hft-algotrade-bot\
├── docker/
│   └── Dockerfile.jupyter         # Jupyter Lab с TA-Lib
├── scripts/
│   └── windows/                   # PowerShell автоматизация
│       ├── deploy.ps1             # Полное развертывание
│       ├── backtest.ps1           # Запуск бэктестов
│       ├── download-data.ps1      # Загрузка данных
│       └── logs.ps1               # Просмотр логов
├── user_data/
│   ├── config/
│   │   └── config.json            # Главный конфиг
│   ├── strategies/                # Торговые стратегии
│   │   ├── SimpleTestStrategy.py  # По умолчанию
│   │   ├── StoicStrategyV1.py     # Продвинутая
│   │   └── ...
│   ├── data/binance/              # Исторические данные
│   ├── logs/                      # Логи приложения
│   │   └── freqtrade.log
│   └── tradesv3.sqlite            # База сделок
├── research/                      # Jupyter ноутбуки
│   ├── 01_strategy_template.ipynb
│   └── README.md
├── docker-compose.yml             # Конфигурация сервисов
├── QUICKSTART.md                  # Этот файл
├── STRUCTURE.md                   # Детальная структура
└── LOGS.md                        # Гайд по логам
```

---

## 🔍 Мониторинг и логи

### Просмотр логов в реальном времени

```powershell
# Все логи Freqtrade
docker-compose logs -f freqtrade

# Последние 100 строк
docker-compose logs -f --tail=100 freqtrade

# Логи всех сервисов
docker-compose logs -f

# Файловые логи
cat .\user_data\logs\freqtrade.log

# Фильтр по ERROR
Get-Content .\user_data\logs\freqtrade.log | Select-String "ERROR"
```

### Health checks

```powershell
# Проверить здоровье контейнеров
docker-compose ps

# Детальная инспекция
docker inspect stoic_freqtrade

# Проверить API
curl http://localhost:8080/api/v1/ping

# Dashboard доступность
curl http://localhost:3000
```

---

## ⚙️ Конфигурация

### Основные параметры (user_data/config/config.json)

```json
{
  "dry_run": true,              // Бумажная торговля (НЕ реальные деньги)
  "dry_run_wallet": 10000,      // Виртуальный кошелек: 10000 USDT
  "max_open_trades": 3,         // Максимум открытых позиций
  "stake_currency": "USDT",     // Валюта стейка
  "timeframe": "5m",            // Таймфрейм свечей
  "exchange": {
    "name": "binance",          // Биржа
    "key": "",                  // API ключ (не нужен для dry_run)
    "secret": ""                // API secret
  },
  "pair_whitelist": [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT"
  ],
  "stoploss": -0.05            // Стоплосс: -5%
}
```

### Переключение на реальную торговлю

⚠️ **ВНИМАНИЕ**: Только после тщательного тестирования!

```json
{
  "dry_run": false,             // Отключить бумажную торговлю
  "exchange": {
    "name": "binance",
    "key": "ваш_api_key",       // Добавить реальные ключи
    "secret": "ваш_api_secret"
  }
}
```

---

## 🐛 Устранение неполадок

### Проблема: Контейнер постоянно перезапускается

```powershell
# Посмотреть логи с самого начала
docker-compose logs freqtrade

# Частые причины:
# 1. Стратегия не найдена
#    Решение: Проверить имя стратегии в docker-compose.yml

# 2. Конфиг содержит ошибки
#    Решение: Проверить синтаксис config.json

# 3. Недостаточно RAM
#    Решение: Закрыть другие приложения, увеличить лимиты Docker
```

### Проблема: "Config file not found" при backtesting

```powershell
# НЕПРАВИЛЬНО:
docker-compose run --rm freqtrade backtesting --strategy SimpleTestStrategy

# ПРАВИЛЬНО (указать полный путь к конфигу):
docker-compose run --rm freqtrade backtesting `
  --config /freqtrade/user_data/config/config.json `
  --strategy SimpleTestStrategy
```

### Проблема: "Could not import strategy" или "No module named"

```powershell
# Причина: Стратегия имеет ошибки импорта или не существует

# Решение 1: Использовать SimpleTestStrategy (гарантированно работает)
docker-compose exec freqtrade ls /freqtrade/user_data/strategies/

# Решение 2: Проверить стратегию на ошибки
docker-compose exec freqtrade python -c "import sys; sys.path.insert(0, '/freqtrade/user_data/strategies'); from SimpleTestStrategy import SimpleTestStrategy"
```

### Проблема: Долгое скачивание данных

```powershell
# Уменьшить период:
docker-compose run --rm freqtrade download-data `
  --config /freqtrade/user_data/config/config.json `
  --days 30  # Вместо 90

# Или уменьшить количество пар:
  --pairs BTC/USDT ETH/USDT  # Только 2 пары
```

### Проблема: FreqUI не подключается к API

```powershell
# Проверить API доступность
curl http://localhost:8080/api/v1/ping

# Должен вернуть: {"status":"pong"}

# Если не отвечает:
# 1. Убедиться что Freqtrade запущен и здоров
docker-compose ps

# 2. Проверить environment variables в docker-compose.yml
#    FREQTRADE__API_SERVER__ENABLED=true

# 3. Перезапустить сервисы
docker-compose down
docker-compose up -d freqtrade frequi
```

### Проблема: Jupyter build падает

```powershell
# Если долго строится (5-10 минут нормально):
docker-compose build --no-cache jupyter

# Если ошибки с TA-Lib:
# Уже исправлено - компиляция из исходников

# Если ошибки с pandas-ta:
# Убедитесь что версия 0.3.14 (без 'b')
```

---

## 📈 Следующие шаги

### 1. Изучить стратегии

```powershell
# Открыть стратегию в редакторе
code .\user_data\strategies\SimpleTestStrategy.py

# Или в Jupyter Lab
# http://localhost:8888 → strategies/SimpleTestStrategy.py
```

### 2. Создать свою стратегию

```powershell
# Скопировать шаблон
copy .\user_data\strategies\SimpleTestStrategy.py .\user_data\strategies\MyStrategy.py

# Изменить класс:
class MyStrategy(IStrategy):
    # Ваша логика
```

### 3. Оптимизировать параметры (HyperOpt)

```powershell
# Загрузить больше данных
docker-compose run --rm freqtrade download-data `
  --config /freqtrade/user_data/config/config.json `
  --days 180

# Запустить HyperOpt (100 эпох)
docker-compose run --rm freqtrade hyperopt `
  --config /freqtrade/user_data/config/config.json `
  --hyperopt-loss SharpeHyperOptLoss `
  --strategy StoicStrategyV1 `
  --epochs 100 `
  --spaces buy sell roi stoploss
```

### 4. Анализ результатов

```powershell
# Plotting (требует установленного freqtrade локально)
freqtrade plot-dataframe `
  --strategy SimpleTestStrategy `
  --timerange 20241101-20241201

# Или через Jupyter Lab:
# Открыть research/01_strategy_template.ipynb
```

---

## 📚 Дополнительные ресурсы

- **Официальная документация Freqtrade**: https://www.freqtrade.io/en/stable/
- **GitHub Issues**: https://github.com/kandibobe/hft-algotrade-bot/issues
- **STRUCTURE.md**: Детальное описание структуры проекта
- **LOGS.md**: Продвинутый гайд по логам и отладке

---

## 🆘 Поддержка

Если возникли проблемы:

1. **Проверьте LOGS.md** - там описаны частые ошибки
2. **Посмотрите логи**: `docker-compose logs -f freqtrade`
3. **Проверьте статус**: `docker-compose ps`
4. **GitHub Issues**: Создайте issue с описанием проблемы и логами

---

**Удачной торговли! 🚀📈**
