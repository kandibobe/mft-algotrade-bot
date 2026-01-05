# Пайплайн Обучения ML

Полный MLOps пайплайн для обучения, отслеживания и развертывания моделей машинного обучения.

## Возможности

### 1. Инжиниринг Признаков (Feature Engineering)

Преобразование сырых данных OHLCV в готовые для ML признаки:

- **50+ Технических Индикаторов** - Цена, объем, моментум, волатильность, тренд.
- **Временные Признаки** - Циклическое кодирование часов, дней, месяцев.
- **Масштабирование (Scaling)** - StandardScaler, MinMaxScaler, RobustScaler.
- **Удаление Корреляции** - Автоматическое обнаружение и удаление избыточных признаков.
- **Настраиваемый Пайплайн** - Гибкий выбор групп признаков.

### 2. Обучение Моделей

Обучение и оптимизация ML-моделей с подбором гиперпараметров:

- **Поддерживаемые Модели** - Random Forest, XGBoost, LightGBM.
- **Оптимизация Гиперпараметров** - Автоматический тюнинг на базе Optuna.
- **Кросс-валидация** - Разбиение с учетом временных рядов (Time-series aware).
- **Отбор Признаков** - Автоматическое ранжирование важности признаков.
- **Сохранение Моделей** - Сериализация обученных моделей.

### 3. Трекинг Экспериментов

Отслеживание экспериментов через W&B или MLflow:

- **Логирование Гиперпараметров** - Сохранение всех конфигураций.
- **Метрики** - Метрики обучения, валидации и бэктестов.
- **Артефакты** - Сохранение моделей, графиков, данных.
- **Сравнение** - Сравнение нескольких запусков.

### 4. Реестр Моделей (Model Registry)

Управление версиями и развертыванием:

- **Версионирование** - Отслеживание нескольких версий модели.
- **Процесс Валидации** - Проверка перед продакшеном.
- **Продвижение (Promotion)** - Безопасный деплой в прод.
- **Откат (Rollback)** - Возврат к предыдущим версиям.

## Быстрый Старт

### Базовый Инжиниринг Признаков

```python
from src.ml.training import FeatureEngineer
import pandas as pd

# Загрузка данных
df = pd.read_csv("ohlcv_data.csv", parse_dates=['date'], index_col='date')

# Создание инженера
engineer = FeatureEngineer()

# Трансформация
features_df = engineer.transform(df)

print(f"Сгенерировано {len(engineer.get_feature_names())} признаков")
```

### Обучение Модели

```python
from src.ml.training import ModelTrainer, ModelConfig

# Подготовка данных
X_train = features_df[engineer.get_feature_names()]
y_train = (features_df['close'].shift(-1) > features_df['close']).astype(int)

# Конфигурация
config = ModelConfig(
    model_type="random_forest",
    optimize_hyperparams=True,
    n_trials=50,
)

# Обучение
trainer = ModelTrainer(config)
model, metrics = trainer.train(X_train, y_train)

print(f"F1 Score: {metrics['f1']:.4f}")
```

## Полный Рабочий Процесс (Workflow)

### End-to-End ML Pipeline

1.  **Загрузка данных**: Получение исторических свечей.
2.  **Генерация признаков**: Создание технических индикаторов.
3.  **Разделение данных**: Train/Validation split (без заглядывания в будущее!).
4.  **Трекинг**: Запуск эксперимента в W&B.
5.  **Обучение**: Запуск `ModelTrainer` с оптимизацией Optuna.
6.  **Логирование**: Сохранение метрик и важности признаков.
7.  **Регистрация**: Добавление модели в `ModelRegistry`.
8.  **Валидация**: Проверка метрик на соответствие минимальным требованиям.
9.  **Деплой**: Продвижение модели в статус `PRODUCTION`.

## Интеграция с Freqtrade

### Использование ML в Стратегии

```python
from freqtrade.strategy import IStrategy
from src.ml.training import ModelRegistry
import pickle

class MLStrategy(IStrategy):
    def __init__(self, config: dict):
        super().__init__(config)

        # Загрузка продакшен модели
        registry = ModelRegistry()
        prod_model = registry.get_production_model("trend_classifier")

        if prod_model:
            with open(prod_model.model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.feature_names = prod_model.feature_names
```

## Лучшие Практики

1.  **Всегда используйте Time-Series Split**: Никогда не используйте случайное перемешивание (shuffle) при разделении данных на обучение и тест. Это приведет к утечке данных из будущего (Lookahead Bias).
2.  **Валидируйте перед Деплоем**: Используйте `registry.validate_model()` с жесткими порогами метрик.
3.  **Следите за Drift-ом**: Если метрики живой торговли начинают сильно отличаться от метрик обучения, необходимо переобучить модель (Retrain).
