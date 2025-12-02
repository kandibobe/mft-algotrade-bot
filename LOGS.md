# 📋 Stoic Citadel - Руководство по логам и отладке

Комплексный гайд по мониторингу, диагностике и решению проблем в HFT боте.

---

## 📍 Где находятся логи?

### Docker логи (в памяти контейнеров)

```powershell
# Просмотр логов Freqtrade
docker-compose logs freqtrade

# Следование за логами в реальном времени
docker-compose logs -f freqtrade

# Последние 100 строк
docker-compose logs --tail=100 freqtrade

# С временными метками
docker-compose logs --timestamps freqtrade

# Все сервисы
docker-compose logs -f
```

### Файловые логи (постоянное хранение)

**Расположение**: `user_data/logs/freqtrade.log`

```powershell
# Просмотр последних 50 строк
Get-Content .\user_data\logs\freqtrade.log -Tail 50

# Следование за логами
Get-Content .\user_data\logs\freqtrade.log -Wait

# Поиск ошибок
Get-Content .\user_data\logs\freqtrade.log | Select-String "ERROR"

# Поиск по паттерну
Get-Content .\user_data\logs\freqtrade.log | Select-String "Strategy"

# Экспорт ошибок в файл
Get-Content .\user_data\logs\freqtrade.log | Select-String "ERROR" > errors.txt
```

---

## 🎯 Использование PowerShell скриптов

### Скрипт logs.ps1

**Базовое использование**:

```powershell
# Последние 50 строк Freqtrade
.\scripts\windows\logs.ps1

# Последние 100 строк
.\scripts\windows\logs.ps1 -Lines 100

# Следование за логами
.\scripts\windows\logs.ps1 -Follow

# Другой сервис
.\scripts\windows\logs.ps1 -Service frequi -Lines 30
```

**Фильтрация**:

```powershell
# Только ошибки
.\scripts\windows\logs.ps1 -Level ERROR

# Только предупреждения
.\scripts\windows\logs.ps1 -Level WARNING

# Поиск по тексту
.\scripts\windows\logs.ps1 -Search "Strategy"

# Комбинация
.\scripts\windows\logs.ps1 -Level ERROR -Lines 200
```

**Файловые логи**:

```powershell
# Просмотр freqtrade.log
.\scripts\windows\logs.ps1 -FileLog

# С фильтром по уровню
.\scripts\windows\logs.ps1 -FileLog -Level ERROR

# Экспорт в файл
.\scripts\windows\logs.ps1 -FileLog -Level ERROR -Export
```

**Все сервисы**:

```powershell
# Логи всех контейнеров
.\scripts\windows\logs.ps1 -Service all -Lines 50
```

---

## 🔍 Типовые сообщения и что они означают

### ✅ Успешные операции (INFO)

```
freqtrade.worker - INFO - Starting worker 2024.11
```
**Значение**: Freqtrade запускается нормально

```
freqtrade.exchange.exchange - INFO - Using Exchange "Binance"
```
**Значение**: Подключение к бирже установлено

```
freqtrade.rpc.telegram - INFO - Telegram is listening for following commands
```
**Значение**: Telegram бот активен и готов к командам

```
freqtrade.strategy.interface - INFO - Strategy 'SimpleTestStrategy' successfully loaded
```
**Значение**: Стратегия загружена успешно

### ⚠️ Предупреждения (WARNING)

```
freqtrade.resolvers.iresolver - WARNING - Could not import /freqtrade/user_data/strategies/StoicCitadelV2.py
```
**Причина**: Стратегия содержит ошибки импорта  
**Решение**: Проверить импорты в стратегии или использовать SimpleTestStrategy

```
freqtrade.exchange.exchange - WARNING - Pair BTC/USDT not available
```
**Причина**: Пара недоступна на бирже или неверное название  
**Решение**: Проверить список пар в config.json

```
freqtrade.persistence.models - WARNING - Trade using more than 1x stake amount
```
**Причина**: Активная сделка использует больше стейка чем обычно  
**Решение**: Проверить параметр `stake_amount` в конфиге

### ❌ Ошибки (ERROR)

```
freqtrade - ERROR - Impossible to load Strategy 'StoicStrategyV1'. This class does not exist or contains Python code errors.
```
**Причина**: Стратегия не найдена или содержит ошибки Python  
**Решение**:
1. Убедиться что файл существует: `docker-compose exec freqtrade ls /freqtrade/user_data/strategies/`
2. Проверить имя класса в файле стратегии
3. Использовать SimpleTestStrategy как fallback

```
freqtrade - ERROR - Configuration error: DEPRECATED: Setting 'protections' in the configuration is deprecated.
```
**Причина**: Устаревшая конфигурация в config.json  
**Решение**: Удалить секцию `"protections"` из user_data/config/config.json

```
freqtrade.exchange.exchange - ERROR - DDosProtection: binance GET https://api.binance.com/api/v3/exchangeInfo 429
```
**Причина**: Превышен лимит запросов к API биржи  
**Решение**: Подождать 1-2 минуты, биржа автоматически разблокирует

```
freqtrade.persistence.models - ERROR - Unable to create trade with stake_amount=0
```
**Причина**: Недостаточно средств или неверная конфигурация  
**Решение**: Проверить `dry_run_wallet` в config.json

---

## 🚨 Распространенные проблемы и решения

### Проблема 1: Контейнер постоянно перезапускается

**Симптомы**:
```
stoic_freqtrade exited with code 2 (restarting)
```

**Диагностика**:
```powershell
# Посмотреть логи с самого начала
docker-compose logs freqtrade | Select-String "ERROR"

# Проверить статус
docker-compose ps
```

**Частые причины**:

1. **Стратегия не найдена**
   ```powershell
   # Проверить доступные стратегии
   docker-compose exec freqtrade ls /freqtrade/user_data/strategies/
   
   # Изменить стратегию в docker-compose.yml на SimpleTestStrategy
   ```

2. **Ошибки в config.json**
   ```powershell
   # Валидация JSON
   Get-Content .\user_data\config\config.json | ConvertFrom-Json
   
   # Если ошибка - исправить синтаксис JSON
   ```

3. **Недостаточно RAM**
   ```powershell
   # Проверить использование памяти
   docker stats --no-stream
   
   # Увеличить лимиты в Docker Desktop Settings
   ```

### Проблема 2: "Config file not found" при backtesting

**Ошибка**:
```
ERROR - Config file "config.json" not found!
```

**Причина**: Не указан полный путь к конфигу

**Решение**:
```powershell
# НЕПРАВИЛЬНО:
docker-compose run --rm freqtrade backtesting --strategy SimpleTestStrategy

# ПРАВИЛЬНО:
docker-compose run --rm freqtrade backtesting `
  --config /freqtrade/user_data/config/config.json `
  --strategy SimpleTestStrategy
```

### Проблема 3: FreqUI не подключается к API

**Симптомы**:
- FreqUI показывает "Connection failed"
- API не отвечает на http://localhost:8080

**Диагностика**:
```powershell
# Проверить API
curl http://localhost:8080/api/v1/ping

# Должен вернуть: {"status":"pong"}

# Проверить логи FreqUI
docker-compose logs frequi
```

**Решения**:

1. **API не запущен**
   ```powershell
   # Проверить environment variables в docker-compose.yml
   # FREQTRADE__API_SERVER__ENABLED=true
   
   # Перезапустить
   docker-compose restart freqtrade
   ```

2. **Неверные credentials**
   ```yaml
   # В docker-compose.yml должно быть:
   - FREQTRADE__API_SERVER__USERNAME=stoic_admin
   - FREQTRADE__API_SERVER__PASSWORD=StoicGuard2024
   ```

3. **Freqtrade не здоров**
   ```powershell
   docker-compose ps
   # Если status не "healthy" - смотреть логи
   docker-compose logs freqtrade
   ```

### Проблема 4: Долгая загрузка стратегии

**Симптомы**:
```
Starting worker 2024.11
[30 секунд тишины]
Strategy loaded
```

**Причины**:
- Стратегия загружает большие datasets
- Медленные библиотеки (TA-Lib, ML models)

**Решение**:
```python
# В стратегии использовать ленивую загрузку:
def __init__(self, config: dict) -> None:
    super().__init__(config)
    self.model = None  # Не загружать сразу

def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    if self.model is None:
        self.model = self.load_model()  # Загрузить при первом использовании
```

### Проблема 5: "No module named 'signals.indicators'"

**Ошибка**:
```
WARNING - Could not import /freqtrade/user_data/strategies/StoicCitadelV2.py 
due to 'No module named 'signals.indicators'; 'signals' is not a package'
```

**Причина**: Неправильная структура импортов в стратегии

**Решение**:
1. **Использовать SimpleTestStrategy** (гарантированно работает)
2. **Исправить импорты** в проблемной стратегии:
   ```python
   # Заменить относительные импорты на абсолютные
   # Было:
   from signals.indicators import custom_indicator
   
   # Стало:
   from user_data.strategies.signals.indicators import custom_indicator
   ```

---

## 📊 Мониторинг производительности

### Проверка статуса контейнеров

```powershell
# Быстрая проверка
docker-compose ps

# Детальная статистика
docker stats --no-stream

# Health check конкретного контейнера
docker inspect stoic_freqtrade --format='{{.State.Health.Status}}'
```

### Использование ресурсов

```powershell
# Память и CPU всех контейнеров
docker stats

# Только Freqtrade
docker stats stoic_freqtrade

# Экспорт статистики
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" > stats.txt
```

### Дисковое пространство

```powershell
# Размер Docker images
docker images | Select-String "freqtrade"

# Размер данных
Get-ChildItem -Path .\user_data\data\ -Recurse | Measure-Object -Property Length -Sum

# Очистка неиспользуемых данных
docker system prune -a
```

---

## 🔧 Продвинутая диагностика

### Вход в контейнер

```powershell
# Bash в Freqtrade
docker-compose exec freqtrade bash

# После входа можно:
ls /freqtrade/user_data/strategies/
cat /freqtrade/user_data/config/config.json
python -c "from user_data.strategies.SimpleTestStrategy import SimpleTestStrategy"
```

### Проверка Python зависимостей

```powershell
# Список установленных пакетов
docker-compose exec freqtrade pip list

# Проверка конкретного пакета
docker-compose exec freqtrade pip show freqtrade

# Версии ключевых библиотек
docker-compose exec freqtrade python -c "import freqtrade; print(freqtrade.__version__)"
```

### Проверка подключения к бирже

```powershell
# Тест API Binance
docker-compose exec freqtrade python -c "
import ccxt
exchange = ccxt.binance()
markets = exchange.load_markets()
print(f'Connected! Available pairs: {len(markets)}')
"
```

### Дебаг стратегии

```powershell
# Dry-run тест стратегии
docker-compose run --rm freqtrade test-strategy \
  --config /freqtrade/user_data/config/config.json \
  --strategy SimpleTestStrategy

# Проверка синтаксиса стратегии
docker-compose exec freqtrade python -m py_compile /freqtrade/user_data/strategies/SimpleTestStrategy.py
```

---

## 📈 Анализ торговых логов

### Поиск сигналов входа/выхода

```powershell
# Поиск BUY сигналов
Get-Content .\user_data\logs\freqtrade.log | Select-String "Buy signal found"

# Поиск SELL сигналов
Get-Content .\user_data\logs\freqtrade.log | Select-String "Sell signal found"

# Экспорт в файл для анализа
Get-Content .\user_data\logs\freqtrade.log | Select-String "signal found" > signals.txt
```

### Анализ прибыльности

```powershell
# Поиск закрытых сделок
Get-Content .\user_data\logs\freqtrade.log | Select-String "Selling.*profit"

# Убыточные сделки
Get-Content .\user_data\logs\freqtrade.log | Select-String "Selling.*loss"
```

### Ротация логов

```powershell
# Архивирование старых логов
$date = Get-Date -Format "yyyyMMdd"
Copy-Item .\user_data\logs\freqtrade.log ".\user_data\logs\freqtrade_$date.log"

# Очистка текущего лога
Clear-Content .\user_data\logs\freqtrade.log
```

---

## 🆘 Когда обращаться за помощью

### Подготовка информации для issue

1. **Соберите логи**:
   ```powershell
   # Экспорт всех логов
   docker-compose logs > full_logs.txt
   
   # Только ошибки
   .\scripts\windows\logs.ps1 -Level ERROR -Export
   ```

2. **Информация о системе**:
   ```powershell
   # Версии
   docker --version > system_info.txt
   docker-compose --version >> system_info.txt
   git --version >> system_info.txt
   
   # Конфигурация (удалите API ключи!)
   Get-Content .\user_data\config\config.json >> system_info.txt
   ```

3. **Статус контейнеров**:
   ```powershell
   docker-compose ps > container_status.txt
   docker stats --no-stream >> container_status.txt
   ```

4. **Создайте GitHub Issue** с:
   - Описанием проблемы
   - Шагами для воспроизведения
   - Прикрепленными логами
   - Информацией о системе

---

## 💡 Лучшие практики

1. **Регулярно проверяйте логи**:
   ```powershell
   .\scripts\windows\logs.ps1 -Level WARNING -Lines 100
   ```

2. **Мониторьте здоровье**:
   ```powershell
   docker-compose ps  # Каждые 30 минут
   ```

3. **Архивируйте важные логи**:
   ```powershell
   # Еженедельный бэкап
   $week = Get-Date -UFormat "%V"
   Copy-Item .\user_data\logs\freqtrade.log ".\backups\logs\week_$week.log"
   ```

4. **Используйте уровни логирования**:
   ```json
   // В config.json
   {
     "logging": {
       "level": "INFO"  // DEBUG для детального анализа
     }
   }
   ```

---

## 📚 Дополнительные ресурсы

- **QUICKSTART.md**: Быстрый старт и основные команды
- **STRUCTURE.md**: Описание структуры проекта
- **Официальная документация Freqtrade**: https://www.freqtrade.io/en/stable/
- **Discord сообщество**: https://discord.gg/freqtrade

---

**Успешного мониторинга! 🚀📊**
