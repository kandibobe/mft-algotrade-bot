# 📋 Stoic Citadel - Логи и отладка

## 🎯 Где находятся логи

### 1. Docker контейнеры (stdout/stderr)

```powershell
# Все логи Freqtrade
docker-compose logs freqtrade

# Последние 100 строк
docker-compose logs --tail=100 freqtrade

# Следить в реальном времени
docker-compose logs -f freqtrade

# Логи с временными метками
docker-compose logs -f -t freqtrade

# Логи всех сервисов
docker-compose logs -f
```

### 2. Файловые логи

```powershell
# Основной лог Freqtrade
cat .\user_data\logs\freqtrade.log

# Последние 100 строк
Get-Content .\user_data\logs\freqtrade.log -Tail 100

# Следить в реальном времени
Get-Content .\user_data\logs\freqtrade.log -Wait -Tail 50

# Фильтр по ERROR
Get-Content .\user_data\logs\freqtrade.log | Select-String "ERROR"

# Фильтр по WARNING
Get-Content .\user_data\logs\freqtrade.log | Select-String "WARNING|ERROR"
```

---

## 📊 Уровни логирования

| Уровень | Описание | Пример |
|---------|----------|--------|
| `INFO` | Нормальная работа | "Starting worker", "Trade opened" |
| `WARNING` | Предупреждения | "Could not import strategy", "Rate limit" |
| `ERROR` | Ошибки | "Impossible to load Strategy", "Connection failed" |
| `CRITICAL` | Критичные сбои | "Cannot connect to exchange", "Database corrupted" |

### Изменить уровень логирования:

**В config.json**:
```json
{
  "verbosity": 0,  // 0=INFO, 1=DEBUG, 2=TRACE (очень подробно)
}
```

**Через docker-compose.yml**:
```yaml
command: >
  trade
  --verbosity 1  # DEBUG уровень
```

---

## 🔍 Частые сообщения и их значение

### ✅ Нормальная работа (INFO)

```
2025-12-02 13:48:18 - freqtrade - INFO - freqtrade 2024.11
```
**Что это**: Версия Freqtrade при старте
**Действие**: Нормально

```
2025-12-02 13:48:18 - freqtrade.worker - INFO - Starting worker 2024.11
```
**Что это**: Рабочий процесс запустился
**Действие**: Нормально

```
2025-12-02 13:48:18 - freqtrade.configuration.configuration - INFO - Runmode set to dry_run.
```
**Что это**: Режим бумажной торговли активен
**Действие**: Нормально (безопасный режим)

```
2025-12-02 13:48:18 - freqtrade.exchange.check_exchange - INFO - Exchange "binance" is officially supported
```
**Что это**: Биржа Binance поддерживается официально
**Действие**: Нормально

```
2025-12-02 13:52:01 - freqtrade.data.history.history_utils - INFO - Download history data for "BTC/USDT"
```
**Что это**: Загружаются исторические данные
**Действие**: Нормально, подождите завершения

---

### ⚠️ Предупреждения (WARNING)

```
WARNING - Could not import /freqtrade/user_data/strategies/StoicCitadelV2.py due to 'No module named ...'
```
**Что это**: Стратегия имеет ошибки импорта  
**Причина**: Отсутствует модуль или синтаксическая ошибка  
**Действие**: 
- Если используете эту стратегию → исправить импорты
- Если НЕ используете → игнорировать (не влияет на работу)

```
time="2025-12-02T14:42:30+01:00" level=warning msg="docker-compose.yml: the attribute `version` is obsolete"
```
**Что это**: Docker Compose предупреждает о deprecated атрибуте  
**Причина**: `version: '3.8'` устарел  
**Действие**: Удалить первую строку из docker-compose.yml (уже исправлено)

---

### ❌ Ошибки (ERROR)

```
ERROR - Impossible to load Strategy 'StoicStrategyV1'. This class does not exist or contains Python code errors.
```
**Что это**: Стратегия не может быть загружена  
**Причины**:
1. Файл стратегии не существует
2. Имя класса не совпадает
3. Синтаксическая ошибка в коде
4. Отсутствуют зависимости

**Решение**:
```powershell
# Проверить наличие файла
docker-compose exec freqtrade ls /freqtrade/user_data/strategies/

# Проверить содержимое
docker-compose exec freqtrade cat /freqtrade/user_data/strategies/StoicStrategyV1.py | Select-String "class"

# Тест импорта
docker-compose exec freqtrade python -c "from user_data.strategies.StoicStrategyV1 import StoicStrategyV1"

# Переключиться на рабочую стратегию
# Редактировать docker-compose.yml:
  --strategy SimpleTestStrategy  # <- Использовать SimpleTestStrategy
```

```
ERROR - Config file "config.json" not found!
```
**Что это**: Конфиг не найден при запуске команды  
**Причина**: `docker-compose run` не использует правильную рабочую директорию  
**Решение**: Всегда указывать полный путь

```powershell
# НЕПРАВИЛЬНО:
docker-compose run --rm freqtrade backtesting --strategy SimpleTestStrategy

# ПРАВИЛЬНО:
docker-compose run --rm freqtrade backtesting `
  --config /freqtrade/user_data/config/config.json `
  --strategy SimpleTestStrategy
```

```
ERROR - Configuration error: DEPRECATED: Setting 'protections' in the configuration is deprecated.
```
**Что это**: Секция `protections` устарела в Freqtrade 2024.11  
**Решение**: Удалить секцию из config.json (уже исправлено)

---

### 🔥 Критичные ошибки (CRITICAL)

```
CRITICAL - Cannot connect to exchange 'binance'
```
**Что это**: Не удается подключиться к бирже  
**Причины**:
1. Нет интернета
2. Binance недоступен
3. API ключи неверны (для live режима)

**Решение**:
```powershell
# Проверить интернет
Test-Connection -ComputerName www.binance.com -Count 4

# Проверить статус Binance
curl https://api.binance.com/api/v3/ping

# Должен вернуть: {}
```

---

## 🔧 Диагностика проблем

### Проблема: Контейнер постоянно перезапускается

```powershell
# 1. Проверить статус
docker-compose ps

# Если видите "Restarting" или "Exit 1/2":

# 2. Посмотреть полные логи
docker-compose logs freqtrade

# 3. Посмотреть последние 50 строк перед крашем
docker-compose logs --tail=50 freqtrade

# 4. Инспекция контейнера
docker inspect stoic_freqtrade
```

**Частые причины**:

| Симптом | Причина | Решение |
|---------|---------|--------|
| `ERROR - Impossible to load Strategy` | Стратегия не найдена | Использовать SimpleTestStrategy |
| `ERROR - Config file not found` | Неправильный путь к config | Указать `/freqtrade/user_data/config/config.json` |
| `CRITICAL - Cannot connect to exchange` | Нет интернета | Проверить подключение |
| Exit code 137 | Недостаточно RAM | Увеличить лимиты Docker |

### Проблема: API недоступен (FreqUI не подключается)

```powershell
# 1. Проверить, что Freqtrade запущен
docker-compose ps

# Должен быть "Up" и "healthy"

# 2. Проверить API напрямую
curl http://localhost:8080/api/v1/ping

# Должен вернуть: {"status":"pong"}

# 3. Если не отвечает - посмотреть логи API
docker-compose logs freqtrade | Select-String "API"

# 4. Проверить environment variables
docker-compose config | Select-String "API"
```

### Проблема: Данные не загружаются / долго грузятся

```powershell
# 1. Проверить доступность Binance
curl https://api.binance.com/api/v3/exchangeInfo

# 2. Посмотреть прогресс загрузки
docker-compose logs -f freqtrade

# Поиск:
# "Downloaded data for BTC/USDT with length 26087" - успешно
# "Rate limit exceeded" - слишком много запросов, подождать

# 3. Уменьшить нагрузку
# Загружать меньше дней:
  --days 30  # Вместо 90

# Или меньше пар:
  --pairs BTC/USDT ETH/USDT  # Только 2 пары
```

### Проблема: Бэктест падает с ошибкой

```powershell
# 1. Проверить наличие данных
docker-compose exec freqtrade ls -lh /freqtrade/user_data/data/binance/

# Должны быть файлы *.feather

# 2. Проверить временной диапазон
# Убедиться что timerange соответствует доступным данным
# Например, если данных с 2024-09-03:
  --timerange 20240903-  # Правильно
  --timerange 20240801-  # Неправильно - нет данных за август

# 3. Запустить с --dry-run-wallet (если ошибка с балансом)
docker-compose run --rm freqtrade backtesting `
  --config /freqtrade/user_data/config/config.json `
  --strategy SimpleTestStrategy `
  --dry-run-wallet 10000
```

---

## 📈 Мониторинг производительности

### CPU и память контейнеров

```powershell
# Использование ресурсов в реальном времени
docker stats

# Для конкретного контейнера
docker stats stoic_freqtrade

# Лимиты и использование
docker inspect stoic_freqtrade | Select-String "Memory"
```

### Размер логов

```powershell
# Размер файлового лога
Get-ChildItem .\user_data\logs\freqtrade.log | Select-Object Name, Length

# Если лог очень большой (>100MB), ротировать:
move .\user_data\logs\freqtrade.log .\user_data\logs\freqtrade_$(Get-Date -Format 'yyyyMMdd').log.old
```

### Health checks

```powershell
# Статус всех сервисов
docker-compose ps

# Детальная информация о health
docker inspect stoic_freqtrade --format='{{json .State.Health}}' | ConvertFrom-Json

# API health
curl http://localhost:8080/api/v1/ping
curl http://localhost:8080/api/v1/show_config
```

---

## 🛠️ Продвинутая отладка

### Войти в контейнер

```powershell
# Bash в Freqtrade контейнере
docker-compose exec freqtrade bash

# Теперь внутри:
cd /freqtrade/user_data
ls -la
python -c "from strategies.SimpleTestStrategy import SimpleTestStrategy; print('OK')"
```

### Проверить версии библиотек

```powershell
docker-compose exec freqtrade pip list | Select-String "freqtrade|ccxt|pandas"
```

### Тестировать стратегию без запуска бота

```powershell
# Сухой прогон (dry-run test)
docker-compose exec freqtrade python -c "
import sys
sys.path.insert(0, '/freqtrade/user_data/strategies')
from SimpleTestStrategy import SimpleTestStrategy
s = SimpleTestStrategy()
print('Strategy loaded successfully!')
print(f'Timeframe: {s.timeframe}')
print(f'Stoploss: {s.stoploss}')
"
```

### Проверить конфиг на валидность

```powershell
# JSON валидация
docker-compose exec freqtrade python -c "
import json
with open('/freqtrade/user_data/config/config.json') as f:
    config = json.load(f)
print('Config valid!')
print(f'Strategy: {config.get(\"strategy\", \"not set\")}')
"
```

---

## 📋 Cheat Sheet

### Быстрые команды для копипасты

```powershell
# === ПРОСМОТР ЛОГОВ ===
docker-compose logs -f --tail=100 freqtrade
Get-Content .\user_data\logs\freqtrade.log -Wait -Tail 50

# === ФИЛЬТРЫ ===
docker-compose logs freqtrade | Select-String "ERROR|WARNING"
Get-Content .\user_data\logs\freqtrade.log | Select-String "ERROR" | Select-Object -Last 20

# === ДИАГНОСТИКА ===
docker-compose ps
docker stats stoic_freqtrade --no-stream
curl http://localhost:8080/api/v1/ping

# === РЕСТАРТ ===
docker-compose restart freqtrade
docker-compose down && docker-compose up -d freqtrade frequi

# === ОЧИСТКА ===
docker-compose down
docker system prune -af --volumes

# === BACKUP ЛОГОВ ===
move .\user_data\logs\freqtrade.log .\user_data\logs\backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').log
```

---

## 🆘 Когда обращаться за помощью

Если после проверки всех логов и диагностики проблема не решена:

1. **Соберите информацию**:
   ```powershell
   # Версия Docker
   docker --version
   docker-compose --version
   
   # Логи контейнеров
   docker-compose logs --tail=200 > logs_output.txt
   
   # Конфигурация
   docker-compose config > compose_config.txt
   
   # Статус
   docker-compose ps > containers_status.txt
   ```

2. **Создайте GitHub Issue**:
   - URL: https://github.com/kandibobe/hft-algotrade-bot/issues
   - Приложите:
     - Описание проблемы
     - Шаги для воспроизведения
     - Логи (logs_output.txt)
     - Версии ПО
     - Скриншоты (если применимо)

---

**Удачной отладки! 🔧🐛**
