# Быстрый старт (5 минут)

## 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

## 2. Настройка окружения
```bash
cp .env.example .env
# Отредактируй .env - добавь API ключи биржи
```

## 3. Скачать данные
```bash
python scripts/download_data.py BTC/USDT --timeframe 1h --days 365
```

## 4. Запустить backtest
```bash
python scripts/run_backtest.py --strategy StoicEnsembleStrategyV3 --timerange 20230101-20241231
```

## 5. Запустить paper trading
```bash
# Редактировать config/paper_trading_config.yaml
freqtrade trade --config config/paper_trading_config.yaml
```

## Troubleshooting
- Ошибка импорта? → `pip install -r requirements.txt`
- Нет данных? → `python scripts/download_data.py`
