# 🚀 БЫСТРЫЙ СТАРТ

**3 шага до запуска:**

```bash
# 1️⃣ Клонируй и запусти setup
git clone https://github.com/kandibobe/hft-algotrade-bot.git
cd hft-algotrade-bot
make setup

# 2️⃣ Скачай данные (опционально, можно пропустить)
make download

# 3️⃣ Запусти бота в тестовом режиме
make trade-dry
```

**Готово!** Открой: http://localhost:3000

---

## 🎯 Что происходит?

- **Freqtrade** - торговый бот (тестовый режим, без реальных денег)
- **FreqUI** - дашборд на http://localhost:3000
- **Jupyter Lab** - для исследований на http://localhost:8888 (token: stoic2024)

---

## 📊 Основные команды

```bash
make start        # Запустить всё
make stop         # Остановить
make logs         # Посмотреть логи
make test         # Запустить тесты
make backtest     # Бэктест стратегии
```

---

## ⚙️ Настройка API (для реальной торговли)

1. Открой `.env`
2. Добавь свои API ключи от биржи
3. Измени `dry_run: false` в `user_data/config/config.json`
4. Запусти: `make trade-live` ⚠️

---

## 🔬 Разработка стратегий

```bash
# Запусти Jupyter
make research

# Открой http://localhost:8888 (token: stoic2024)
# Загрузи research/01_strategy_template.ipynb
```

---

## 🆘 Проблемы?

**Docker не запускается:**
```bash
docker-compose down
docker-compose build --no-cache
make start
```

**Нет данных:**
```bash
make download
```

**Ошибки в логах:**
```bash
make logs SERVICE=freqtrade
```

---

## 📚 Больше информации

- [README.md](README.md) - полная документация
- [user_data/strategies/](user_data/strategies/) - примеры стратегий
- [research/](research/) - исследовательские ноутбуки

---

**🏛️ Stoic Citadel - Where reason rules, not emotion.**
