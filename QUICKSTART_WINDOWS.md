# 🚀 Quick Start Guide - Windows

Step-by-step guide для запуска на Windows.

---

## 📋 Предварительные требования

✅ Docker Desktop установлен и запущен
✅ Python 3.11+ установлен
✅ Git установлен

---

## 🏁 STEP 1: Запуск контейнеров

```powershell
# В PowerShell
cd C:\hft-algotrade-bot

# Запустить все контейнеры
docker-compose up -d

# Проверить статус
docker ps
```

**Ожидаемый результат:**
```
stoic_freqtrade   - Торговый бот (Running)
stoic_frequi      - Web UI (Running)
stoic_postgres    - База данных (Healthy)
stoic_redis       - Кэш (Healthy)
```

---

## 📥 STEP 2: Скачать данные

### Вариант A: Через Docker (рекомендуется)

```powershell
# Одной строкой! (обрати внимание на backtick ` вместо \)
docker exec stoic_freqtrade freqtrade download-data `
  --exchange binance `
  --timeframe 5m `
  --pairs BTC/USDT `
  --days 30
```

### Вариант B: Через Python скрипт

```powershell
python scripts/download_data.py --preset major --days 30
```

**Пресеты:**
- `major` - BTC, ETH, BNB (3 пары)
- `layer1` - SOL, AVAX, NEAR, ADA (4 пары)
- `all` - Все популярные (12 пар)

---

## 🔄 STEP 3: Синхронизировать данные

После скачивания данных, скопируй их из Docker:

```powershell
# Запустить скрипт синхронизации
.\scripts\sync_data.ps1
```

**Или вручную:**
```powershell
docker cp stoic_freqtrade:/freqtrade/user_data/data/binance/. user_data/data/binance/
```

---

## 🔍 STEP 4: Проверить данные

```powershell
# Посмотреть список файлов
dir user_data\data\binance\

# Инспектировать данные
python scripts/inspect_data.py --pair BTC/USDT

# Сравнить несколько пар
python scripts/inspect_data.py --compare BTC/USDT ETH/USDT
```

**Ожидаемый output:**
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
   ...
```

---

## 🎯 STEP 5: Запустить backtest

### Quick Test (7 дней)

```powershell
python scripts/run_backtest.py --profile quick
```

### Full Test (30 дней)

```powershell
python scripts/run_backtest.py --profile full
```

### Custom Test

```powershell
python scripts/run_backtest.py `
  --pair BTC/USDT ETH/USDT `
  --days 14 `
  --timeframe 5m
```

---

## 🌐 STEP 6: Открыть FreqUI

1. **Открой браузер:** http://localhost:3000

2. **Войди:**
   - Username: `stoic_admin`
   - Password: `StoicGuard2024!ChangeMe`

3. **Посмотри результаты:**
   - Dashboard - Текущее состояние бота
   - Backtesting - Результаты бэктестов
   - Trades - История сделок

---

## 🔧 Troubleshooting

### Проблема: "Data not found"

**Решение:**
```powershell
# 1. Скачай данные
docker exec stoic_freqtrade freqtrade download-data `
  --exchange binance `
  --timeframe 5m `
  --pairs BTC/USDT `
  --days 30

# 2. Синхронизируй
.\scripts\sync_data.ps1

# 3. Проверь
dir user_data\data\binance\
```

### Проблема: "Docker container not found"

**Решение:**
```powershell
# Проверь статус
docker ps -a

# Запусти контейнеры
docker-compose up -d

# Проверь логи
docker logs stoic_freqtrade --tail 50
```

### Проблема: PowerShell multiline commands не работают

**Причина:** В PowerShell используется `` ` `` (backtick), а не `\` (backslash)

**Правильно:**
```powershell
docker exec stoic_freqtrade freqtrade download-data `
  --exchange binance `
  --pairs BTC/USDT
```

**Неправильно:**
```powershell
# ❌ НЕ работает в PowerShell!
docker exec stoic_freqtrade freqtrade download-data \
  --exchange binance \
  --pairs BTC/USDT
```

### Проблема: "Module not found"

**Решение:**
```powershell
# Активируй virtual environment
.\.venv\Scripts\Activate.ps1

# Установи зависимости
pip install -r requirements.txt
```

---

## 🎉 Готово!

Теперь у тебя есть:

✅ Запущенный торговый бот (в dry-run режиме)
✅ Скачанные исторические данные
✅ Работающий backtest engine
✅ Доступ к FreqUI для мониторинга

---

## 📚 Следующие шаги

1. **Изучи результаты backtest в FreqUI**
   - Посмотри на win rate, profit factor, drawdown

2. **Попробуй разные профили:**
   ```powershell
   python scripts/run_backtest.py --list-profiles
   python scripts/run_backtest.py --profile aggressive
   ```

3. **Настрой стратегию:**
   - Отредактируй параметры в `user_data/config/config_production.json`
   - Измени take_profit, stop_loss в config

4. **Запусти paper trading:**
   - Убедись что `dry_run = true` в config
   - Мониторь логи: `docker logs stoic_freqtrade -f`

5. **НИКОГДА не запускай live trading без:**
   - Минимум 2 недели paper trading
   - Положительные результаты на бэктестах
   - Понимания всех рисков

---

## 🆘 Нужна помощь?

- [TOOLS_GUIDE.md](docs/TOOLS_GUIDE.md) - Полная документация по всем утилитам
- [TESTING_GUIDE.md](docs/TESTING_GUIDE.md) - Руководство по тестированию
- [CREDENTIALS.md](CREDENTIALS.md) - Все пароли и доступы
- [GitHub Issues](https://github.com/kandibobe/mft-algotrade-bot/issues)

---

**Last Updated:** 2025-12-19
**Version:** 2.0 - Windows Edition
