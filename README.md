# 🏛️ Stoic Citadel

**Professional HFT-lite Algorithmic Trading Infrastructure**

[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Freqtrade](https://img.shields.io/badge/Freqtrade-Powered-orange?style=flat-square)](https://www.freqtrade.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> *"In research, we seek truth. In trading, we execute truth."*

---

## 🚀 Quick Start

**3 команды до запуска:**

```bash
git clone https://github.com/kandibobe/hft-algotrade-bot.git
cd hft-algotrade-bot
make setup && make trade-dry
```

**Готово!** Открой http://localhost:3000

📖 Подробнее: [START.md](START.md)

---

## 🎯 Философия

**Stoic Citadel** отделяет исследование от исполнения:

- 🔬 **Research Lab** (Jupyter + VectorBT) - где ты ищешь edge
- ⚡ **Execution Engine** (Freqtrade) - где ты деплоишь проверенные стратегии

**Принципы:**
1. Research First - стратегии в лабе, не в продакшене
2. Risk Management - сохранение капитала > максимизация прибыли
3. Automation - машина исполняет, человек исследует
4. Discipline - никакой мести рынку, никаких эмоций

---

## 🏗️ Архитектура

```
┌─────────────────────────────────────────────┐
│           STOIC CITADEL                     │
├─────────────────────────────────────────────┤
│                                             │
│  Research Lab          Execution Engine     │
│  ─────────────        ────────────────      │
│  • Jupyter Lab    ──► • Freqtrade           │
│  • VectorBT           • FreqUI              │
│  • ML Models          • WebSocket API       │
│  • Backtesting        • Order Execution     │
│                                             │
│  Infrastructure                             │
│  ──────────────                             │
│  • PostgreSQL  • Telegram Bot               │
│  • Prometheus  • Grafana  • Portainer       │
│                                             │
└─────────────────────────────────────────────┘
```

| Компонент | Назначение | Порт |
|-----------|------------|------|
| **Freqtrade** | Торговый бот | 8080 |
| **FreqUI** | Web дашборд | 3000 |
| **Jupyter Lab** | Исследования | 8888 |
| **PostgreSQL** | Аналитика | 5432 |
| **Prometheus** | Метрики | 9090 |
| **Grafana** | Визуализация | 3001 |
| **Portainer** | Docker UI | 9000 |

---

## ✨ Возможности

### 🔬 Research Lab
- Быстрый бэктестинг с VectorBT
- 50+ готовых индикаторов
- ML pipeline (XGBoost, LightGBM, CatBoost)
- Интерактивные графики (Plotly)
- Walk-forward validation

### ⚡ Execution Engine
- Низкая латентность (<1 сек)
- Риск-менеджмент (hard stops, cooldowns, max drawdown)
- Telegram уведомления
- Мультибиржа (Binance, Bybit, ...)
- Полное логирование в PostgreSQL

### 🐳 Инфраструктура
- Полностью в Docker
- Безопасность (зашифрованные API ключи)
- Запуск одной командой
- 24/7 работа
- Prometheus + Grafana мониторинг

---

## 📦 Требования

- **Docker** >= 20.10
- **Docker Compose** >= 2.0
- **8GB RAM** (минимум)
- **20GB диск** (для данных)

---

## 💻 Основные команды

```bash
# Управление
make start           # Запустить всё
make stop            # Остановить
make logs            # Посмотреть логи

# Разработка
make research        # Jupyter Lab
make test            # Запустить тесты
make lint            # Проверить код

# Торговля
make trade-dry       # Тест (без денег)
make backtest        # Бэктест стратегии
make trade-live      # LIVE ⚠️

# Данные
make download        # Скачать исторические данные
make verify          # Проверить качество данных
```

📖 Все команды: `make help`

---

## ⚙️ Конфигурация

### Биржа

Отредактируй `user_data/config/config.json`:

```json
{
  "exchange": {
    "name": "binance",
    "key": "YOUR_API_KEY",
    "secret": "YOUR_API_SECRET"
  }
}
```

### Telegram

1. Создай бота: [@BotFather](https://t.me/botfather)
2. Узнай chat ID: [@userinfobot](https://t.me/userinfobot)
3. Обнови `.env`:

```env
TELEGRAM_ENABLED=true
TELEGRAM_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789
```

### Риски

Критические настройки в `config.json`:

```json
{
  "max_open_trades": 3,
  "stoploss": -0.05,
  "trailing_stop": true,
  "protections": [...]
}
```

---

## 🔬 Разработка стратегий

### Рабочий процесс

1. **Исследование** → Jupyter Lab (`make research`)
2. **Бэктест** → VectorBT / Freqtrade
3. **Валидация** → Walk-forward тестирование
4. **Имплементация** → `user_data/strategies/`
5. **Тестирование** → Dry-run 1-2 недели
6. **Деплой** → Live (с малым капиталом)

### Пример

```python
# user_data/strategies/MyStrategy.py
from freqtrade.strategy import IStrategy
import talib.abstract as ta

class MyStrategy(IStrategy):
    def populate_indicators(self, dataframe, metadata):
        dataframe['rsi'] = ta.RSI(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        dataframe['enter_long'] = (dataframe['rsi'] < 30)
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        dataframe['exit_long'] = (dataframe['rsi'] > 70)
        return dataframe
```

---

## 🧪 Тестирование

```bash
make test              # Все тесты
make test-unit         # Unit тесты
make test-integration  # Интеграция
make test-coverage     # С покрытием
```

**CI/CD:** Автоматические тесты при каждом push.

---

## 📊 Мониторинг

```bash
make monitoring  # Запустить Grafana + Prometheus
```

Доступ:
- **Grafana**: http://localhost:3001 (admin/admin)
- **Prometheus**: http://localhost:9090

---

## 🛡️ Риск-менеджмент

| Защита | Цель | Настройка |
|--------|------|-----------|
| Hard Stoploss | Ограничить убытки | `stoploss: -0.05` |
| Trailing Stop | Зафиксировать прибыль | `trailing_stop: true` |
| Stoploss Guard | Предотвратить revenge trading | После 3 лоссов |
| Max Drawdown | Circuit breaker | При 15% просадке |
| Cooldown | Принудительный перерыв | 2-4 часа после лоссов |

**Экстренная остановка:**
```bash
make stop
```

---

## 📁 Структура проекта

```
hft-algotrade-bot/
├── research/                  # 🔬 Jupyter ноутбуки
├── user_data/
│   ├── config/
│   │   └── config.json       # ⚙️ Единая конфигурация
│   └── strategies/           # 📈 Торговые стратегии
├── scripts/                  # 🛠️ Автоматизация
├── tests/                    # 🧪 Тесты
├── monitoring/               # 📊 Grafana + Prometheus
├── docker/                   # 🐳 Dockerfiles
├── START.md                  # 🚀 Быстрый старт
└── README.md                 # 📖 Этот файл
```

---

## 🔧 Troubleshooting

### Контейнер не запускается
```bash
make logs SERVICE=freqtrade
docker-compose build --no-cache
make start
```

### Нет данных
```bash
make download
make verify
```

### Ошибки в стратегии
```bash
make test
make backtest STRATEGY=MyStrategy
```

---

## ⚠️ Disclaimer

**ВАЖНО:**

- ⚠️ Это ПО только для **образовательных целей**
- 💰 Торговля криптовалютами несёт **значительный риск**
- 📉 **Прошлые результаты не гарантируют будущих**
- 💸 Ты можешь **потерять весь капитал**
- 🚫 Авторы **не несут ответственности** за твои убытки
- ✅ **Всегда тестируй** в dry-run режиме сначала
- 💵 **Никогда не инвестируй** больше чем можешь потерять

---

## 📄 Лицензия

MIT License - см. [LICENSE](LICENSE)

---

## 🤝 Контакты

- 🐛 Issues: [GitHub Issues](https://github.com/kandibobe/hft-algotrade-bot/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/kandibobe/hft-algotrade-bot/discussions)

---

<p align="center">
  <strong>Built with discipline. Traded with wisdom. Executed with precision.</strong>
  <br><br>
  <em>"The wise trader knows that the best trade is often no trade at all."</em>
  <br><br>
  🏛️ <strong>Stoic Citadel</strong> - Where reason rules, not emotion.
</p>
