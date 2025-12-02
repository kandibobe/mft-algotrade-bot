# 🪟 PowerShell Scripts - Automation for Windows

Набор PowerShell скриптов для автоматизации работы со Stoic Citadel на Windows.

---

## 📋 Список скриптов

| Скрипт | Назначение | Основные параметры |
|--------|------------|-------------------|
| **deploy.ps1** | Полное развертывание проекта | `-SkipData`, `-WithJupyter`, `-AllServices` |
| **backtest.ps1** | Запуск бэктестов | `-Strategy`, `-Timerange`, `-ExportTrades` |
| **download-data.ps1** | Загрузка исторических данных | `-Days`, `-Timeframe`, `-WithBTC1d` |
| **logs.ps1** | Просмотр и анализ логов | `-Service`, `-Level`, `-Follow` |

---

## 🚀 deploy.ps1 - Полное развертывание

### Описание
Автоматизированный процесс:
1. Проверка системных требований
2. Остановка старых контейнеров
3. Обновление из Git
4. Запуск Docker сервисов
5. Загрузка данных
6. Запуск бэктеста
7. Вывод информации о доступе

### Параметры

```powershell
.\scripts\windows\deploy.ps1 [параметры]
```

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `-SkipData` | Switch | False | Пропустить загрузку данных |
| `-SkipBacktest` | Switch | False | Пропустить бэктест |
| `-WithJupyter` | Switch | False | Запустить Jupyter Lab |
| `-AllServices` | Switch | False | Запустить все сервисы (PostgreSQL, Portainer) |
| `-DataDays` | Int | 90 | Количество дней данных для загрузки |
| `-Strategy` | String | "SimpleTestStrategy" | Стратегия для бэктеста |

### Примеры использования

```powershell
# Базовое развертывание
.\scripts\windows\deploy.ps1

# Без загрузки данных (если уже загружены)
.\scripts\windows\deploy.ps1 -SkipData

# С Jupyter Lab для исследований
.\scripts\windows\deploy.ps1 -WithJupyter

# Все сервисы включая PostgreSQL и Portainer
.\scripts\windows\deploy.ps1 -AllServices

# Только запуск сервисов, без данных и бэктеста
.\scripts\windows\deploy.ps1 -SkipData -SkipBacktest

# 180 дней данных с продвинутой стратегией
.\scripts\windows\deploy.ps1 -DataDays 180 -Strategy "StoicStrategyV1"
```

### Что делает скрипт

1. **Проверки**:
   - Docker Desktop установлен и запущен
   - Docker Compose доступен
   - Git установлен (опционально)

2. **Остановка**:
   - `docker-compose down`

3. **Обновление** (если .git присутствует):
   - `git pull origin simplify-architecture`

4. **Запуск сервисов**:
   - Базовый: `freqtrade` + `frequi`
   - С Jupyter: + `jupyter`
   - Все: + `postgres` + `portainer`

5. **Загрузка данных** (если не `-SkipData`):
   - 90 дней (или `-DataDays`) по 5 парам
   - Таймфрейм: 5m
   - BTC/USDT 1d (для продвинутых стратегий)

6. **Бэктест** (если не `-SkipBacktest`):
   - Последние 2 месяца
   - Выбранная стратегия

7. **Итог**:
   - Ссылки на сервисы
   - Credentials
   - Полезные команды

---

## 📊 backtest.ps1 - Запуск бэктестов

### Описание
Гибкий запуск бэктестов с настройкой параметров и экспортом результатов.

### Параметры

```powershell
.\scripts\windows\backtest.ps1 [параметры]
```

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `-Strategy` | String | "SimpleTestStrategy" | Имя стратегии |
| `-Timerange` | String | Авто (60 дней) | Период YYYYMMDD-YYYYMMDD |
| `-StartDaysAgo` | Int | 60 | Дней назад от сегодня |
| `-Pairs` | String | Из config.json | Торговые пары через пробел |
| `-MaxOpenTrades` | Int | 3 | Максимум открытых позиций |
| `-EnablePositionStacking` | Switch | False | Стекинг позиций |
| `-ExportTrades` | Switch | False | Экспорт результатов в JSON |
| `-Breakdown` | Switch | False | Детализация (день/неделя/месяц) |

### Примеры использования

```powershell
# Быстрый тест SimpleTestStrategy
.\scripts\windows\backtest.ps1

# Другая стратегия
.\scripts\windows\backtest.ps1 -Strategy "StoicStrategyV1"

# Конкретный период
.\scripts\windows\backtest.ps1 -Strategy "SimpleTestStrategy" -Timerange "20241001-20241201"

# Последние 90 дней
.\scripts\windows\backtest.ps1 -StartDaysAgo 90

# С экспортом результатов
.\scripts\windows\backtest.ps1 -Strategy "StoicStrategyV1" -ExportTrades

# С детальной разбивкой по времени
.\scripts\windows\backtest.ps1 -Breakdown

# Больше открытых позиций
.\scripts\windows\backtest.ps1 -MaxOpenTrades 5

# Только BTC и ETH
.\scripts\windows\backtest.ps1 -Pairs "BTC/USDT ETH/USDT"

# Полная комбинация
.\scripts\windows\backtest.ps1 `
  -Strategy "StoicStrategyV1" `
  -Timerange "20240101-20241201" `
  -MaxOpenTrades 5 `
  -ExportTrades `
  -Breakdown
```

### Экспорт результатов

Если используется `-ExportTrades`, результаты сохраняются в:
```
user_data/backtest_results/backtest-result-YYYYMMDD-HHMMSS.json
```

Формат JSON содержит:
- Общую статистику
- Детали каждой сделки
- Метрики по парам
- Sharpe ratio, Sortino, Max drawdown

---

## 💾 download-data.ps1 - Загрузка данных

### Описание
Загрузка исторических данных с биржи для бэктестинга и анализа.

### Параметры

```powershell
.\scripts\windows\download-data.ps1 [параметры]
```

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `-Days` | Int | 90 | Количество дней данных |
| `-Timeframe` | String | "5m" | Таймфрейм (1m, 5m, 15m, 1h, 4h, 1d) |
| `-Exchange` | String | "binance" | Биржа |
| `-Pairs` | String | BTC/ETH/BNB/SOL/XRP | Торговые пары через пробел |
| `-WithBTC1d` | Switch | False | Загрузить BTC 1d для фильтра |
| `-BTC1dDays` | Int | 365 | Дней BTC 1d данных |
| `-TradingViewFormat` | Switch | False | Формат TradingView JSON |

### Примеры использования

```powershell
# Стандартная загрузка (90 дней, 5m, 5 пар)
.\scripts\windows\download-data.ps1

# Больше данных
.\scripts\windows\download-data.ps1 -Days 180

# Часовой таймфрейм
.\scripts\windows\download-data.ps1 -Timeframe "1h"

# Годовые данные на дневном таймфрейме
.\scripts\windows\download-data.ps1 -Days 365 -Timeframe "1d"

# С BTC 1d для продвинутых стратегий
.\scripts\windows\download-data.ps1 -WithBTC1d

# Только определенные пары
.\scripts\windows\download-data.ps1 -Pairs "BTC/USDT ETH/USDT"

# Другая биржа
.\scripts\windows\download-data.ps1 -Exchange "kraken"

# TradingView формат
.\scripts\windows\download-data.ps1 -TradingViewFormat

# Полная комбинация
.\scripts\windows\download-data.ps1 `
  -Days 180 `
  -Timeframe "5m" `
  -Pairs "BTC/USDT ETH/USDT BNB/USDT" `
  -WithBTC1d `
  -BTC1dDays 500
```

### Оценка времени и размера

Скрипт показывает оценки:
- Количество пар
- Примерный размер (~0.5MB на пару в день)
- Примерное время загрузки (~3 сек на пару)

### Проверка данных

После загрузки скрипт показывает:
- Количество загруженных файлов
- Общий размер
- Список файлов с размерами

---

## 📋 logs.ps1 - Просмотр логов

### Описание
Универсальный инструмент для просмотра, фильтрации и экспорта логов.

### Параметры

```powershell
.\scripts\windows\logs.ps1 [параметры]
```

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `-Service` | String | "freqtrade" | Сервис (freqtrade/frequi/jupyter/all) |
| `-Lines` | Int | 50 | Количество строк |
| `-Follow` | Switch | False | Следование за логами (live) |
| `-Timestamps` | Switch | False | Показать временные метки |
| `-Level` | String | "" | Фильтр ERROR/WARNING/INFO/DEBUG |
| `-Search` | String | "" | Поиск по тексту |
| `-FileLog` | Switch | False | Файловые логи (freqtrade.log) |
| `-Export` | Switch | False | Экспорт в текстовый файл |

### Примеры использования

```powershell
# Последние 50 строк Freqtrade
.\scripts\windows\logs.ps1

# Больше строк
.\scripts\windows\logs.ps1 -Lines 200

# Следование за логами в реальном времени
.\scripts\windows\logs.ps1 -Follow

# Другой сервис
.\scripts\windows\logs.ps1 -Service frequi

# Все сервисы
.\scripts\windows\logs.ps1 -Service all

# Только ошибки
.\scripts\windows\logs.ps1 -Level ERROR

# Только предупреждения
.\scripts\windows\logs.ps1 -Level WARNING

# Поиск по тексту
.\scripts\windows\logs.ps1 -Search "Strategy"

# Файловые логи Freqtrade
.\scripts\windows\logs.ps1 -FileLog

# Файловые логи с фильтром
.\scripts\windows\logs.ps1 -FileLog -Level ERROR

# Экспорт логов в файл
.\scripts\windows\logs.ps1 -Level ERROR -Export

# Комбинации
.\scripts\windows\logs.ps1 -Level WARNING -Lines 100 -Timestamps
.\scripts\windows\logs.ps1 -FileLog -Search "BTC/USDT" -Export
```

### Цветовой вывод

Скрипт автоматически окрашивает логи:
- 🔴 **ERROR** - красный
- 🟡 **WARNING** - желтый
- ⚪ **INFO** - белый
- ⚫ **DEBUG** - серый

### Экспорт логов

При использовании `-Export` создается файл:
```
logs_SERVICE_export_YYYYMMDD_HHMMSS.txt
```

Содержит отфильтрованные логи в чистом текстовом формате.

---

## 🔄 Типичные workflow

### Первичная настройка

```powershell
# 1. Клонировать репозиторий
git clone https://github.com/kandibobe/hft-algotrade-bot.git
cd hft-algotrade-bot

# 2. Переключиться на ветку
git checkout simplify-architecture

# 3. Полное развертывание
.\scripts\windows\deploy.ps1
```

### Ежедневная работа

```powershell
# Проверить статус
docker-compose ps

# Посмотреть логи
.\scripts\windows\logs.ps1 -Follow

# Запустить бэктест новой стратегии
.\scripts\windows\backtest.ps1 -Strategy "MyNewStrategy" -ExportTrades
```

### Обновление данных

```powershell
# Загрузить свежие данные
.\scripts\windows\download-data.ps1 -Days 7

# Запустить бэктест на новых данных
.\scripts\windows\backtest.ps1
```

### Отладка проблем

```powershell
# Проверить ошибки
.\scripts\windows\logs.ps1 -Level ERROR -Lines 200

# Экспортировать для анализа
.\scripts\windows\logs.ps1 -FileLog -Level ERROR -Export

# Просмотреть детальные логи конкретного сервиса
.\scripts\windows\logs.ps1 -Service freqtrade -Follow -Timestamps
```

### Разработка стратегий

```powershell
# Запустить Jupyter
.\scripts\windows\deploy.ps1 -WithJupyter -SkipData -SkipBacktest

# Загрузить данные для анализа
.\scripts\windows\download-data.ps1 -Days 180 -WithBTC1d

# Тестировать стратегию
.\scripts\windows\backtest.ps1 -Strategy "MyStrategy" -Breakdown -ExportTrades
```

---

## 🛠️ Требования

- **PowerShell**: 5.1+ (встроен в Windows)
- **Docker Desktop**: 4.25+
- **Git**: 2.40+ (опционально)
- **Права**: Запуск от имени пользователя (не требуется admin)

---

## 💡 Советы

### Разрешение выполнения скриптов

Если скрипты не запускаются:

```powershell
# Проверить политику
Get-ExecutionPolicy

# Разрешить (временно для текущей сессии)
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# Или разрешить для текущего пользователя
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Автозаполнение параметров

PowerShell поддерживает Tab-completion:

```powershell
.\scripts\windows\backtest.ps1 -St<Tab>  # → -Strategy
.\scripts\windows\logs.ps1 -Se<Tab>      # → -Service
```

### Справка по параметрам

```powershell
Get-Help .\scripts\windows\deploy.ps1 -Detailed
Get-Help .\scripts\windows\backtest.ps1 -Full
```

### Алиасы (опционально)

Создать короткие алиасы:

```powershell
# В PowerShell профиле ($PROFILE)
Set-Alias deploy "C:\hft-algotrade-bot\scripts\windows\deploy.ps1"
Set-Alias backtest "C:\hft-algotrade-bot\scripts\windows\backtest.ps1"
Set-Alias logs "C:\hft-algotrade-bot\scripts\windows\logs.ps1"
Set-Alias dldata "C:\hft-algotrade-bot\scripts\windows\download-data.ps1"

# Использование
deploy -WithJupyter
backtest -Strategy "StoicStrategyV1"
logs -Follow
```

---

## 🐛 Устранение неполадок

### "не является внутренней или внешней командой"

**Проблема**: PowerShell не находит docker или docker-compose

**Решение**:
1. Убедитесь что Docker Desktop запущен
2. Перезапустите PowerShell
3. Проверьте PATH: `$env:PATH -split ';' | Select-String docker`

### "Выполнение скриптов отключено"

**Проблема**: Политика выполнения блокирует скрипты

**Решение**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Скрипт завершается с ошибкой

**Решение**:
1. Запустить с `-Verbose` для детального вывода
2. Проверить логи: `.\scripts\windows\logs.ps1 -Level ERROR`
3. Проверить статус контейнеров: `docker-compose ps`

---

## 📚 Связанная документация

- **QUICKSTART.md**: Быстрый старт проекта
- **LOGS.md**: Детальное руководство по логам
- **STRUCTURE.md**: Описание структуры проекта
- **README.md**: Главная документация

---

**Удачной автоматизации! 🚀🪟**
