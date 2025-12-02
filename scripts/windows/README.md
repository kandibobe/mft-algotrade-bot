# 🪟 PowerShell Scripts для Windows

Пакет автоматизации для Stoic Citadel на Windows.

## 📋 Доступные скрипты

### 1. `deploy.ps1` - Полное развертывание

**Что делает**:
- Подтягивает последние изменения из Git
- Останавливает существующие контейнеры
- Собирает Jupyter Lab (опционально)
- Запускает Freqtrade + FreqUI
- Загружает тестовые данные (опционально)
- Запускает первый бэктест (опционально)

**Использование**:

```powershell
# Полное развертывание (все шаги)
.\scripts\windows\deploy.ps1

# Пропустить сборку Jupyter
.\scripts\windows\deploy.ps1 -SkipJupyter

# Пропустить загрузку данных
.\scripts\windows\deploy.ps1 -SkipData

# Только запуск сервисов (без данных и бэктеста)
.\scripts\windows\deploy.ps1 -SkipData -SkipBacktest
```

---

### 2. `backtest.ps1` - Запуск бэктестов

**Что делает**:
- Запускает бэктест с указанной стратегией
- Поддерживает кастомные параметры

**Использование**:

```powershell
# Базовый бэктест (SimpleTestStrategy)
.\scripts\windows\backtest.ps1

# С указанием стратегии
.\scripts\windows\backtest.ps1 -Strategy "StoicStrategyV1"

# С временным диапазоном
.\scripts\windows\backtest.ps1 -Strategy "SimpleTestStrategy" -Timerange "20241001-20241201"

# С лимитом открытых позиций
.\scripts\windows\backtest.ps1 -MaxOpenTrades 5

# С position stacking
.\scripts\windows\backtest.ps1 -EnablePositionStacking
```

**Параметры**:
- `-Strategy` - Имя стратегии (default: SimpleTestStrategy)
- `-Timerange` - Период бэктеста (default: 20241001-)
- `-Config` - Путь к конфигу (default: /freqtrade/user_data/config/config.json)
- `-MaxOpenTrades` - Лимит позиций
- `-EnablePositionStacking` - Включить усреднение

---

### 3. `download-data.ps1` - Загрузка данных

**Что делает**:
- Загружает исторические данные с Binance
- Сохраняет в user_data/data/binance/

**Использование**:

```powershell
# Загрузить 90 дней 5m данных (по умолчанию)
.\scripts\windows\download-data.ps1

# Указать количество дней
.\scripts\windows\download-data.ps1 -Days 180

# Указать таймфрейм
.\scripts\windows\download-data.ps1 -Timeframe "1h"

# Указать пары
.\scripts\windows\download-data.ps1 -Pairs "BTC/USDT","ETH/USDT"

# Комбинация параметров
.\scripts\windows\download-data.ps1 -Days 365 -Timeframe "1d" -Pairs "BTC/USDT"
```

**Параметры**:
- `-Days` - Количество дней назад (default: 90)
- `-Timeframe` - Таймфрейм свечей (default: 5m)
- `-Pairs` - Массив торговых пар (default: BTC/USDT ETH/USDT BNB/USDT SOL/USDT XRP/USDT)
- `-Exchange` - Биржа (default: binance)
- `-Config` - Путь к конфигу

---

### 4. `logs.ps1` - Просмотр логов

**Что делает**:
- Показывает логи Docker контейнеров
- Фильтрация по уровню
- Режим real-time (follow)

**Использование**:

```powershell
# Последние 100 строк Freqtrade
.\scripts\windows\logs.ps1

# Последние 500 строк
.\scripts\windows\logs.ps1 -Lines 500

# Следить в реальном времени
.\scripts\windows\logs.ps1 -Follow

# Только ERROR
.\scripts\windows\logs.ps1 -Level ERROR

# Только WARNING и ERROR
.\scripts\windows\logs.ps1 -Level WARNING

# Логи другого сервиса
.\scripts\windows\logs.ps1 -Service "frequi"
```

**Параметры**:
- `-Service` - Имя сервиса (default: freqtrade)
- `-Lines` - Количество строк (default: 100)
- `-Follow` - Следить в реальном времени
- `-Level` - Фильтр по уровню (ALL, INFO, WARNING, ERROR)

---

## 🚀 Типовые сценарии

### Первое развертывание

```powershell
# Полная установка с данными и тестом
.\scripts\windows\deploy.ps1
```

### Ежедневный workflow

```powershell
# Обновить данные
.\scripts\windows\download-data.ps1 -Days 1

# Запустить бэктест
.\scripts\windows\backtest.ps1 -Strategy "StoicStrategyV1"

# Посмотреть логи
.\scripts\windows\logs.ps1 -Follow
```

### Разработка стратегии

```powershell
# Загрузить больше данных
.\scripts\windows\download-data.ps1 -Days 180

# Тестировать с разными параметрами
.\scripts\windows\backtest.ps1 -Strategy "MyStrategy" -Timerange "20240101-20240630"
.\scripts\windows\backtest.ps1 -Strategy "MyStrategy" -Timerange "20240701-20241231"

# Мониторить ошибки
.\scripts\windows\logs.ps1 -Level ERROR -Follow
```

### Troubleshooting

```powershell
# Полный рестарт
docker-compose down
.\scripts\windows\deploy.ps1 -SkipData -SkipBacktest

# Проверить логи на ошибки
.\scripts\windows\logs.ps1 -Level ERROR -Lines 500

# Rebuild Jupyter если были проблемы
docker-compose build --no-cache jupyter
```

---

## ⚙️ Требования

- **PowerShell** 5.1 или выше (встроен в Windows 10/11)
- **Docker Desktop** запущен
- **Git** (для deploy.ps1)
- Находиться в корневой директории проекта

---

## 📚 Дополнительная документация

- `QUICKSTART.md` - Основной гайд по быстрому старту
- `LOGS.md` - Детальная информация по логам
- `STRUCTURE.md` - Структура проекта

---

## 🔧 Кастомизация

Все скрипты можно редактировать под свои нужды. Они написаны на чистом PowerShell без внешних зависимостей.

**Пример**: Изменить дефолтные пары для загрузки данных:

```powershell
# Отредактировать download-data.ps1
# Найти строку:
[string[]]$Pairs = @("BTC/USDT", "ETH/USDT", ...)

# Изменить на свои пары:
[string[]]$Pairs = @("BTC/USDT", "LINK/USDT", "AVAX/USDT")
```

---

**Удачной автоматизации! 🚀**
