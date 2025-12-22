# План исправлений для StoicEnsembleStrategyV4

## Текущее состояние
Стратегия V4 имеет следующие проблемы:
1. **ML модель не генерирует сигналы** - порог вероятности 0.65 слишком высок при плотности сигналов 4.8%
2. **Дисбаланс классов** - 95.2% негативных vs 4.8% позитивных сигналов
3. **Неподходящие параметры Triple Barrier** - TP=1.5%, SL=0.75% слишком агрессивны для 5m таймфрейма
4. **Недостаточно данных** - всего 31 день для обучения
5. **Feature engineering требует улучшений** - нет динамических порогов

## Что нужно сделать прямо сейчас (Priority 1)

### 1. Исправить параметры Triple Barrier Labeling
**Файл:** `src/ml/training/labeling.py`
**Изменения:**
- Изменить `take_profit` с 0.015 на 0.008 (0.8%)
- Изменить `stop_loss` с 0.0075 на 0.004 (0.4%)
- Увеличить `max_holding_period` с 24 до 48 (4 часа)
- Добавить динамические барьеры на основе ATR

### 2. Реализовать динамический порог вероятности
**Файл:** `user_data/strategies/StoicEnsembleStrategyV4.py`
**Изменения:**
- Заменить фиксированный порог 0.65 на адаптивный
- Использовать перцентиль предсказаний (например, 75-й перцентиль)
- Минимальный порог 0.55, максимальный 0.75
- Регулировать порог в зависимости от рыночного режима

### 3. Добавить балансировку классов
**Файл:** `src/ml/training/model_trainer.py`
**Изменения:**
- Добавить параметр `class_weight='balanced'` в RandomForest
- Реализовать SMOTE для oversampling minority class
- Добавить параметр `scale_pos_weight` для XGBoost

### 4. Улучшить feature engineering
**Файл:** `src/ml/training/feature_engineering.py`
**Изменения:**
- Добавить больше lag features
- Добавить взаимодействия между индикаторами
- Улучшить обработку пропущенных значений
- Добавить rolling statistics

## План выполнения (Phase 1 - Immediate)

### День 1: Исправление параметров и порогов
1. **Утро:** Обновить параметры Triple Barrier
   - Изменить `TripleBarrierConfig` в labeling.py
   - Протестировать с новыми параметрами
   
2. **День:** Реализовать динамический порог
   - Модифицировать `populate_entry_trend` в V4 стратегии
   - Добавить расчет адаптивного порога
   - Протестировать генерацию сигналов

3. **Вечер:** Балансировка классов
   - Обновить `ModelTrainer._create_model`
   - Добавить SMOTE в training pipeline
   - Переобучить модели

### День 2: Улучшение feature engineering
1. **Утро:** Добавить новые фичи
   - Lag features (1, 2, 3 периода)
   - Rolling statistics (mean, std, min, max)
   - Взаимодействия индикаторов
   
2. **День:** Улучшить валидацию
   - Добавить более строгие проверки на data leakage
   - Улучшить обработку outliers
   - Добавить feature selection

3. **Вечер:** Сбор дополнительных данных
   - Скачать 6+ месяцев исторических данных
   - Добавить multiple timeframes (1h, 4h)
   - Расширить список торговых пар

### День 3: Тестирование и валидация
1. **Утро:** Walk-forward backtest
   - Запустить `scripts/walk_forward_analysis.py`
   - Проверить генерацию сигналов
   - Оценить метрики качества
   
2. **День:** Оптимизация гиперпараметров
   - Запустить hyperparameter optimization
   - Тестировать разные модели (XGBoost, LightGBM)
   - Выбрать лучшую конфигурацию
   
3. **Вечер:** Документация и отчет
   - Обновить документацию
   - Создать отчет о результатах
   - Подготовить план для Phase 2

## Phase 2: Улучшения средней сложности (Week 2)

### 1. Реализовать meta-learning
- Добавить regime detection
- Адаптивные параметры стратегии
- Динамическое взвешивание моделей

### 2. Улучшить risk management
- Динамический position sizing
- Correlation-aware portfolio
- Circuit breakers для экстремальной волатильности

### 3. Создать production pipeline
- Автоматическое переобучение моделей
- Мониторинг дрейфа данных
- A/B тестирование моделей

## Phase 3: Продвинутые улучшения (Month 2-3)

### 1. Расширенный feature engineering
- Order book features
- Sentiment analysis
- On-chain metrics

### 2. Ensemble methods
- Stacking моделей
- Bayesian optimization
- Neural networks для feature extraction

### 3. Multi-timeframe анализ
- Иерархическое моделирование
- Коинтеграция пар
- Портфельная оптимизация

## Технические детали исправлений

### 1. Исправление Triple Barrier параметров
```python
# В файле src/ml/training/labeling.py
@dataclass
class TripleBarrierConfig:
    take_profit: float = 0.008    # Было 0.015
    stop_loss: float = 0.004      # Было 0.0075
    max_holding_period: int = 48  # Было 24
    fee_adjustment: float = 0.001
```

### 2. Динамический порог вероятности
```python
# В файле user_data/strategies/StoicEnsembleStrategyV4.py
def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    # Calculate dynamic threshold based on recent predictions
    if len(dataframe) >= 100:
        recent_predictions = dataframe['ml_prediction'].tail(100)
        # Use 75th percentile, but ensure reasonable bounds
        dynamic_threshold = np.percentile(recent_predictions, 75)
        dynamic_threshold = max(0.55, min(dynamic_threshold, 0.75))
    else:
        dynamic_threshold = 0.6  # Default
    
    # Adjust for regime
    if self._regime_mode == 'defensive':
        dynamic_threshold = max(dynamic_threshold, 0.65)
    elif self._regime_mode == 'aggressive':
        dynamic_threshold = max(0.5, dynamic_threshold * 0.9)
```

### 3. Балансировка классов
```python
# В файле src/ml/training/model_trainer.py
def _create_model(self, **hyperparams) -> Any:
    if self.config.model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        
        # Ensure class_weight is set
        if 'class_weight' not in hyperparams:
            hyperparams['class_weight'] = 'balanced'
            
        return RandomForestClassifier(
            random_state=self.config.random_state,
            n_jobs=-1,
            **hyperparams
        )
```

## Ожидаемые результаты после Phase 1

1. **Генерация сигналов:** 10-20 сделок в месяц (сейчас 0)
2. **Profit Factor:** >1.1 (маржинальная прибыльность)
3. **Sharpe Ratio:** >0.3 (скромные risk-adjusted returns)
4. **Максимальная просадка:** <20%

## Риски и митигации

### Риск 1: Overfitting
**Митигация:** Строгая walk-forward validation, regularization

### Риск 2: Data leakage
**Митигация:** Тщательная проверка feature engineering, lag features

### Риск 3: Regime change
**Митигация:** Meta-learning, адаптивные параметры

## Заключение

План фокусируется на быстрых исправлениях, которые можно реализовать в течение 3 дней, с последующими улучшениями для долгосрочной устойчивости. Ключевые приоритеты:
1. Исправить параметры Triple Barrier
2. Реализовать динамические пороги
3. Балансировка классов
4. Улучшение feature engineering

После реализации этих исправлений стратегия V4 должна начать генерировать сигналы и стать прибыльной.
