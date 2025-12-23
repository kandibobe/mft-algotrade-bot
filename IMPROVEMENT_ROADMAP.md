# ML PIPELINE IMPROVEMENT ROADMAP
**Lead Quantitative AI Researcher Analysis & Master Plan**
**Date:** December 23, 2025
**System:** mft-algotrade-bot

## EXECUTIVE SUMMARY

Current system shows **severe overfitting** (39.2% accuracy drop train→test) and **zero trades executed** due to overly complex conditions. This roadmap outlines a 5-phase systematic approach to achieve **Positive Expectancy** (Precision > 55% on unseen data) with profitable strategy execution.

### CRITICAL ISSUES IDENTIFIED:
1. **Overfitting**: Train accuracy 90.49% vs Test accuracy 51.29%
2. **No Trades**: 8 simultaneous entry conditions too restrictive
3. **Static Barriers**: Fixed TP/SL instead of volatility-adjusted
4. **Feature Noise**: 100+ features without proper selection
5. **Data Leakage**: Potential future data contamination in feature engineering

---

## PHASE 1: DATA & FEATURE HYGIENE (Fixing Garbage In)

### The Concept
Raw prices are non-stationary and confuse ML models. Current feature engineering generates 100+ features with high correlation (>0.95) and potential lookahead bias. We need **stationary features** and **intelligent feature selection** to reduce noise.

### Specific Changes

#### 1.1 Enforce Stationarity (Fractional Differentiation)
**File:** `src/ml/training/feature_engineering.py`
**Logic:** Replace raw price features with fractional differentiation (d=0.5) or log returns. Implement `FractionalDifferentiator` class:
```python
class FractionalDifferentiator:
    def differentiate(self, series: pd.Series, d: float = 0.5) -> pd.Series:
        # Implement fractional differentiation using binomial expansion
        # Returns stationary series preserving memory
```

**Expected Impact:** Reduce spurious correlations by 40%, improve model generalization.

#### 1.2 Advanced Feature Selection (SHAP + Recursive Elimination)
**File:** `src/ml/training/advanced_pipeline.py` (enhance FeatureSelector)
**Logic:** 
1. **Correlation Filter**: Remove features with correlation > 0.85
2. **SHAP Importance**: Train XGBoost on 20% sample, keep top 25 features
3. **Recursive Feature Elimination**: Iteratively remove least important features
4. **Stability Check**: Ensure selected features stable across time folds

**Expected Impact:** Reduce feature count from 100+ to 25-30, increase test precision by 15-20%.

#### 1.3 Fix Data Leakage in Feature Engineering
**File:** `src/ml/training/feature_engineering.py`
**Issues Found:** 
- VWAP uses future data in rolling window
- Rolling statistics without proper lag
- Normalization fitted on entire dataset

**Fix:** Implement strict time-based partitioning:
- Use `expanding_window` instead of `rolling_window` for statistics
- Fit scalers only on training data, never on test/validation
- Add `max_lookback` parameter to prevent future contamination

**Expected Impact:** Eliminate data leakage, reduce overfitting by 25%.

### Success Metrics for Phase 1
- Feature count reduced to ≤ 30
- Feature correlation matrix max < 0.85
- Stationarity confirmed via ADF test (p < 0.05)
- No future data leakage in feature pipeline

---

## PHASE 2: TARGET ENGINEERING (The "Alpha")

### The Concept
Current Triple Barrier Method uses **static barriers** (TP=0.8%, SL=0.4%) that don't adapt to market volatility. During high volatility, stops get hit too often; during low volatility, profit targets are missed. We need **volatility-adjusted dynamic barriers**.

### Specific Changes

#### 2.1 Implement ATR-Based Dynamic Barriers
**File:** `src/ml/training/labeling.py` (enhance DynamicBarrierLabeler)
**Logic:** 
```python
def get_dynamic_barriers_atr(df: pd.DataFrame, atr_multiplier_tp: float = 1.5, 
                            atr_multiplier_sl: float = 0.75) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate barriers as multiples of ATR:
    - Take Profit = entry_price ± (ATR * multiplier_tp)
    - Stop Loss = entry_price ± (ATR * multiplier_sl)
    """
    atr = calculate_atr(df['high'], df['low'], df['close'], period=14)
    tp_pct = atr_multiplier_tp * atr / df['close']
    sl_pct = atr_multiplier_sl * atr / df['close']
    
    # Apply bounds: TP between 0.5% and 5%, SL between 0.25% and 3%
    return tp_pct.clip(0.005, 0.05), sl_pct.clip(0.0025, 0.03)
```

#### 2.2 Regime-Aware Barrier Adjustment
**File:** `src/ml/training/labeling.py`
**Logic:** Adjust barrier multipliers based on market regime:
- **High Volatility Regime** (VIX > 30 or ATR > 2%): Wider barriers (TP=2.0x ATR, SL=1.0x ATR)
- **Normal Regime**: Standard barriers (TP=1.5x ATR, SL=0.75x ATR)
- **Low Volatility Regime** (ATR < 0.5%): Tighter barriers (TP=1.0x ATR, SL=0.5x ATR)

#### 2.3 Implement Purged Labeling (De Prado Methodology)
**Issue:** Overlapping labels cause data leakage
**Solution:** Add embargo period after each label:
```python
def create_labels_with_purging(df, tp_pct, sl_pct, max_hold=48, purge_period=5):
    # After labeling a point, skip next 'purge_period' bars
    # Prevents overlapping information
```

### Success Metrics for Phase 2
- Barrier adaptivity: TP/SL vary with volatility (coefficient > 0.7)
- Label balance: Buy signals 15-25% of dataset (not 0% or 50%)
- Purged labels: No overlapping forward-looking windows
- Positive expectancy: Simulated returns > 0 with dynamic barriers

---

## PHASE 3: MODEL TRAINING & REGULARIZATION (Fixing Overfitting)

### The Concept
Current model shows **severe overfitting** (39.2% drop). XGBoost with default parameters is too complex for limited financial data. We need **simplicity-focused hyperparameter optimization** and **strict walk-forward validation**.

### Specific Changes

#### 3.1 Hyperparameter Optimization with Optuna
**File:** Create `src/ml/training/hyperparameter_optimizer.py`
**Search Space for XGBoost (Force Simplicity):**
```python
search_space = {
    'max_depth': trial.suggest_int('max_depth', 3, 7),  # Shallow trees
    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
    'gamma': trial.suggest_float('gamma', 0.1, 1.0),  # High gamma for regularization
    'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),  # High for simplicity
    'subsample': trial.suggest_float('subsample', 0.5, 0.8),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
    'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),  # L1 regularization
    'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 5.0),  # L2 regularization
}
```

**Optimization Metric:** **Precision** (not accuracy) with minimum 55% threshold.

#### 3.2 Strict Purged Walk-Forward Validation
**File:** `src/ml/training/advanced_pipeline.py` (enhance WalkForwardValidator)
**Logic:** 
1. **Time-Based Splits**: Never shuffle time-series data
2. **Purged Gaps**: 5-bar gap between train and test to prevent leakage
3. **Expanding Window**: Train on past, test on future (no lookahead)
4. **Multiple Validation Folds**: 5-fold walk-forward with 60/20/20 splits

**Implementation:**
```python
class PurgedWalkForwardValidator:
    def create_folds(self, X, y, purge_gap=5):
        # Create folds with purge gaps between train and test
        # Ensure no information leakage from test to train
```

#### 3.3 Ensemble of Simple Models
**Strategy:** Instead of one complex model, use ensemble of 3 simple models:
1. **XGBoost** (optimized for precision)
2. **Random Forest** (optimized for feature importance stability)
3. **Logistic Regression** (linear model as baseline)

**Voting:** Weighted average based on out-of-fold performance.

### Success Metrics for Phase 3
- Overfitting reduced: Train-test accuracy gap < 15%
- Precision on validation: > 55% (target: 60%)
- Model stability: Feature importance correlation across folds > 0.7
- Hyperparameter optimization converges within 100 trials

---

## PHASE 4: STRATEGY LOGIC (The Execution)

### The Concept
Current strategy has **8 simultaneous entry conditions** resulting in **zero trades**. We need **dynamic, regime-aware thresholds** that adapt to market conditions.

### Specific Changes

#### 4.1 Implement Dynamic Probability Thresholds
**File:** `user_data/strategies/StoicEnsembleStrategyV4.py`
**Logic:** Replace fixed threshold (0.55) with dynamic thresholds:

```python
def get_dynamic_threshold(self, dataframe, metadata):
    base_threshold = 0.55
    
    # Adjust based on volatility
    atr_pct = dataframe['atr_pct'].iloc[-1]
    if atr_pct > 0.02:  # High volatility
        return 0.65  # Require higher confidence
    elif atr_pct < 0.005:  # Low volatility
        return 0.50  # Can accept lower confidence
    
    # Adjust based on trend
    if dataframe['close'].iloc[-1] > dataframe['ema_200'].iloc[-1]:
        # Bull market - easier for longs
        return base_threshold - 0.05
    else:
        # Bear market - harder for longs
        return base_threshold + 0.05
```

#### 4.2 Regime-Based Position Sizing
**Logic:** Adjust position size based on model confidence and market regime:
```python
def calculate_position_size(self, confidence, regime):
    base_size = 0.1  # 10% of capital
    
    if regime == 'high_volatility':
        return base_size * 0.5  # Half position
    elif regime == 'strong_trend' and confidence > 0.7:
        return base_size * 1.5  # Larger position
    else:
        return base_size
```

#### 4.3 Simplify Entry Conditions
**Current:** 8 conditions must ALL be true
**New:** 3-tier system with progressive relaxation:
1. **Tier 1 (Core)**: ML probability > dynamic threshold + above EMA200
2. **Tier 2 (Confirmation)**: Volume spike (>1.5x average) OR RSI < 40
3. **Tier 3 (Optional)**: MACD positive + ADX > 20

**Entry Logic:** Require Tier 1 + at least one from Tier 2

#### 4.4 Adaptive Stop Loss Based on Model Confidence
**Logic:** Wider stops for high-confidence trades:
```python
def calculate_stop_loss(self, entry_price, confidence, atr):
    base_sl = entry_price * 0.995  # 0.5% default
    
    if confidence > 0.7:
        # High confidence: wider stop (1.5x ATR)
        return entry_price * (1 - 1.5 * atr / entry_price)
    elif confidence < 0.5:
        # Low confidence: tighter stop (0.5x ATR)
        return entry_price * (1 - 0.5 * atr / entry_price)
    else:
        return base_sl
```

### Success Metrics for Phase 4
- Trade frequency: 5-20 trades per month (not 0)
- Position sizing: Varied based on regime/confidence
- Dynamic thresholds: Adapt to market conditions
- Entry condition success rate: > 60% profitable entries

---

## PHASE 5: EVALUATION METRICS

### The Concept
Accuracy is meaningless in trading (market rises 80% of time). We need **trading-specific metrics** that measure real profitability and risk.

### Specific Changes

#### 5.1 Implement Comprehensive Trading Metrics
**File:** `src/ml/training/advanced_pipeline.py` (enhance TradingMetrics)
**Metrics to Track:**
1. **Precision (Entry Accuracy)**: % of buy signals that were profitable
   - **Target:** > 55% (minimum), > 60% (goal)
   
2. **Profit Factor**: Gross Profit / Gross Loss
   - **Target:** > 1.2 (minimum), > 1.5 (goal)
   
3. **Calmar Ratio**: CAGR / Max Drawdown
   - **Target:** > 1.0 (minimum), > 2.0 (goal)
   
4. **Sharpe Ratio**: Risk-adjusted returns
   - **Target:** > 1.0 (minimum), > 1.5 (goal)
   
5. **Maximum Drawdown**: Worst peak-to-trough
   - **Target:** < 20% (maximum), < 15% (goal)
   
6. **Win Rate & Risk/Reward**: Combined analysis
   - **Target:** Win rate > 40% with avg win/avg loss > 1.5

#### 5.2 Walk-Forward Analysis Dashboard
**File:** Create `scripts/walk_forward_analysis_dashboard.py`
**Features:**
- Performance decay over time (check for overfitting)
- Regime-specific performance (bull/bear/sideways)
- Feature importance stability across folds
- Monte Carlo simulation of returns

#### 5.3 Real-time Performance Monitoring
**Integration:** Connect to existing monitoring system
**Alerts:**
- Precision drops below 50% for 10 consecutive trades
- Drawdown exceeds 15%
- Feature importance correlation drops below 0.5
- Model confidence variance increases > 50%

### Success Metrics for Phase 5
- Precision consistently > 55% in walk-forward analysis
- Profit Factor > 1.2 across all market regimes
- Maximum Drawdown < 20% in stress tests
- Positive Calmar and Sharpe ratios
- Performance stable across time (no decay)

---

## IMPLEMENTATION PRIORITY & TIMELINE

### Week 1: Foundation (Phases 1 & 2)
1. **Day 1-2**: Implement fractional differentiation & fix data leakage
2. **Day 3-4**: Implement SHAP-based feature selection
3. **Day 5-7**: Implement ATR-based dynamic barriers & purged labeling

### Week 2: Model Improvement (Phase 3)
1. **Day 8-9**: Implement Optuna hyperparameter optimization
2. **Day 10-11**: Implement purged walk-forward validation
3. **Day 12-14**: Train and validate simplified ensemble model

### Week 3: Strategy Execution (Phase 4)
1. **Day 15-16**: Implement dynamic thresholds & regime detection
2. **Day 17-18**: Simplify entry conditions & adaptive position sizing
3. **Day 19-21**: Backtest and optimize strategy parameters

### Week 4: Evaluation & Deployment (Phase 5)
1. **Day 22-23**: Implement comprehensive trading metrics
2. **Day 24-25**: Create walk-forward analysis dashboard
3. **Day 26-28**: Final integration, stress testing, deployment

## RISK MITIGATION

### Technical Risks:
1. **Data Quality**: Implement data validation pipeline
2. **Overfitting**: Regular monitoring with walk-forward analysis
3. **Market Regime Changes**: Adaptive parameters and regime detection

### Operational Risks:
1. **Execution Slippage**: Test with realistic commission (0.1%)
2. **Model Decay**: Monthly retraining schedule
3. **Black Swan Events**: Circuit breakers and maximum position limits

## EXPECTED OUTCOMES

### Quantitative Targets:
1. **Precision**: 55% → 60%+ (15% improvement)
2. **Overfitting**: 39% drop → <15% drop (60% reduction)
3. **Trade Frequency**: 0 → 10-20 trades/month
4. **Profit Factor**: N/A → >1.2
5. **Maximum Drawdown**: 0% → <20%

### Qualitative Improvements:
1. **Robustness**: Works across market regimes
2. **Explainability**: SHAP values for feature importance
3. **Maintainability**: Modular, tested pipeline
4. **Monitorability**: Real-time performance tracking

---

## NEXT STEPS

**Immediate Action Required:** Which Phase should we execute first?

**Recommendation:** Start with **Phase 1 (Data & Feature Hygiene)** because:
1. **Foundation First**: Garbage in = garbage out. Fixing data issues enables all other improvements.
2. **Quick Wins**: Feature selection can immediately reduce overfitting.
3. **Low Risk**: Changes are isolated to preprocessing, not affecting live trading.
4. **Prerequisite**: Stationary features are required for proper model training.

**Alternative:** If you want to see immediate trading activity, we could start with **Phase 4 (Strategy Logic)** to fix the "zero trades" issue first.

**Please indicate:** Which Phase should we execute first?
1. Phase 1: Data & Feature Hygiene (Recommended)
2. Phase 2: Target Engineering
3. Phase 3: Model Training & Regularization
4. Phase 4: Strategy Logic
5. Phase 5: Evaluation Metrics

Once you select the starting phase, I
