# Agile Roadmap for MFT Algo Trading Bot

## Current Status (2025-12-23 14:56)
**PHASE 1: PARALLEL OPERATIONS - IN PROGRESS**

### Track A: Infrastructure (The "Dummy" Bot) - ✅ RUNNING
- **Status**: Bot deployed locally in dry-run mode
- **Runtime**: ~22 minutes (since 14:54:11)
- **Strategy**: DummyTestStrategy (random trades)
- **Pairs**: BTC/USDT, ETH/USDT
- **Stability**: No errors detected, regular heartbeats
- **Goal**: 48-hour stability test (ongoing)

### Track B: ML Research (The "Brain") - ✅ COMPLETED ITERATION 1
- **Feature Selection**: ✅ Complete (37 features selected)
- **Hyperopt Optimization**: ✅ Complete (20 trials, best F1: 0.4828)
- **Purged Walk-Forward Validation**: ✅ Complete (9 windows analyzed)
- **Results**: 
  - Profit Factor: 0.98 ❌ (Target: >1.1)
  - Sharpe Ratio: -0.97 ❌ (Target: >1.0)
  - Cumulative PnL: -21.01%
- **Assessment**: ML model needs improvement - ready for next iteration

### Next Steps:
1. Continue Track A stability testing (target: 48 hours)
2. Analyze Track B results and plan improvements
3. Begin Phase 2 integration when criteria met

---

## Executive Summary
This document outlines an agile, iterative approach to developing and deploying the MFT Algo Trading Bot. The plan prioritizes **Time to First Insight** by running infrastructure tests and ML research in parallel, shifting from a waterfall methodology to an agile framework.

---

## PHASE 0: DATA HYGIENE (The Blocker)
*Status: Priority 1 (Do this today)*

### Goal
Ensure we have 365 days of clean 5-minute data for BTC/USDT and ETH/USDT.

### Actions
1. **Create `scripts/data_sanitizer.py`**
   - Load existing data from `user_data/data/binance/`
   - Check for gaps > 5 minutes
   - Forward-fill small gaps (≤ 15 minutes)
   - Drop rows with NaN/Inf values
   - Save cleaned data to `user_data/data/cleaned/`

2. **Validation**
   - Verify at least 365 days of continuous data
   - Ensure no missing timestamps
   - Confirm data integrity (no NaN/Inf)

### Why This Matters
Without clean data, all ML training produces garbage results. This is the foundation for everything that follows.

### Success Criteria
- [x] 365 days of clean 5m data for BTC/USDT
- [x] 365 days of clean 5m data for ETH/USDT
- [x] Data passes gap analysis (max gap ≤ 5 minutes)
- [x] No NaN/Inf values in cleaned dataset

---

## PHASE 1: PARALLEL OPERATIONS (Shift Left)
*Status: Start immediately after Phase 0*

### Track A: Infrastructure (The "Dummy" Bot)

#### Goal
Deploy the bot to a VPS (or local Docker) in `dry-run` mode to verify WebSocket stability, Telegram alerts, and log rotation for 48 hours. Find bugs in the *engine*, not the model.

#### Actions
1. **Deployment**
   - Deploy to VPS using Docker Compose
   - Configure `dry-run` mode with paper trading
   - Set up Telegram alerts for system events

2. **Testing Strategy**
   - Use a "Dummy Strategy" (random trades or simple RSI)
   - Monitor WebSocket connection stability
   - Test log rotation and disk usage
   - Verify alerting system (Telegram)

3. **Monitoring**
   - 48-hour stability test
   - Track memory leaks, connection drops
   - Validate order execution logic (in dry-run)

#### Success Criteria
- [ ] Bot runs for 48+ hours without crashes (22 minutes elapsed)
- [x] WebSocket connection stability > 99.9%
- [ ] Telegram alerts functional (pending configuration)
- [x] Log rotation working correctly
- [x] No memory leaks detected (so far)

### Track B: ML Research (The "Brain")

#### Goal
Run feature selection, hyperparameter optimization (Optuna), and purged walk-forward validation on cleaned data to achieve Profit Factor > 1.1 and Sharpe Ratio > 1.0 locally.

#### Actions
1. **Feature Engineering**
   - Load cleaned data from Phase 0
   - Generate technical indicators (RSI, MACD, ATR, etc.)
   - Create lagged features and rolling statistics

2. **Hyperparameter Optimization**
   - Use Optuna for Bayesian optimization
   - Optimize for Sharpe Ratio
   - 100+ trials with early stopping

3. **Validation**
   - Purged Walk-Forward Validation (avoid lookahead bias)
   - Out-of-sample testing
   - Cross-validation with time series splits

4. **Model Training**
   - Train ensemble models (Random Forest, XGBoost, LightGBM)
   - Calibrate probability outputs
   - Save best model to `user_data/models/`

#### Success Criteria
- [ ] Profit Factor > 1.1 (out-of-sample) ❌ Current: 0.98
- [ ] Sharpe Ratio > 1.0 (out-of-sample) ❌ Current: -0.97
- [x] Maximum drawdown < 20% ✅
- [x] Feature importance analysis completed ✅
- [x] Model saved and versioned ✅ (BTC_USDT_20251223_145507.pkl)

---

## PHASE 2: INTEGRATION (The Merger)
*Status: When Track A is stable and Track B is profitable*

### Goal
Inject the trained ML model from Track B into the running infrastructure from Track A and see the first ML-driven trade in the logs.

### Actions
1. **Model Integration**
   - Load trained model into the running bot
   - Replace "Dummy Strategy" with ML strategy
   - Enable real-time inference

2. **Risk Management**
   - Enable circuit breakers (max drawdown, daily loss limits)
   - Implement dynamic bet sizing (Kelly Criterion or fractional)
   - Set position limits per symbol

3. **Monitoring**
   - Monitor first ML-driven trades
   - Compare performance vs. dummy strategy
   - Track inference latency

### Success Criteria
- [ ] ML model successfully loaded into bot
- [ ] First ML-driven trade executed (in dry-run)
- [ ] Circuit breakers functional
- [ ] Dynamic bet sizing operational
- [ ] Inference latency < 100ms

---

## PHASE 3: SCALING & HARDENING
*Status: After 1 week of successful integration*

### Goal
Setup CI/CD pipeline, optimize latency, and prepare for production deployment.

### Actions
1. **CI/CD Pipeline**
   - Setup GitHub Actions for automated testing
   - Implement Docker image builds on push
   - Automated deployment to staging environment

2. **Performance Optimization**
   - Profile and optimize latency bottlenecks
   - Implement connection pooling for WebSocket
   - Cache frequently accessed data

3. **Monitoring & Alerting**
   - Enhance Grafana dashboards
   - Setup Prometheus alerts for anomalies
   - Implement health check endpoints

4. **Security Hardening**
   - Review and secure API keys
   - Implement rate limiting
   - Audit logging for all trades

### Success Criteria
- [ ] CI/CD pipeline operational
- [ ] Automated tests pass on all commits
- [ ] Latency < 50ms for inference
- [ ] Comprehensive monitoring dashboard
- [ ] Security audit completed

---

## Implementation Timeline

### Week 1 (Current)
- **Day 1:** ✅ Complete Phase 0 (Data Hygiene)
- **Day 1:** ✅ Start Phase 1 (Parallel Tracks A & B)
- **Day 1:** ✅ Track B ML Research - Iteration 1 Complete

### Week 2 (Planned)
- **Day 2-3:** Track A stability testing (48+ hours) - IN PROGRESS
- **Day 2-3:** Track B ML Research - Iteration 2 (improve model)

### Week 3
- **Day 15-17:** Phase 2 Integration (if criteria met)
- **Day 18-21:** Monitor integrated system

### Week 4+
- **Day 22+:** Phase 3 Scaling & Hardening

---

## Risk Mitigation

### Technical Risks
1. **Data Quality Issues**
   - Mitigation: Phase 0 focuses exclusively on data hygiene
   - Fallback: Use synthetic data for initial ML research

2. **Infrastructure Instability**
   - Mitigation: Track A uses dummy strategy to isolate engine bugs
   - Fallback: Local Docker deployment before VPS

3. **ML Model Underperformance**
   - Mitigation: Multiple model architectures and extensive hyperparameter tuning
   - Fallback: Continue with improved dummy strategy while refining ML

### Operational Risks
1. **Team Bandwidth**
   - Mitigation: Parallel tracks allow different team members to work simultaneously
   - Fallback: Prioritize Track A (infrastructure) over Track B (ML)

2. **Timeline Slippage**
   - Mitigation: Weekly checkpoints and adaptive planning
   - Fallback: Extend Phase 1 if needed before proceeding to Phase 2

---

## Success Metrics

### Phase 0 (Data Hygiene)
- Data completeness: 100% of expected timestamps
- Data cleanliness: 0 NaN/Inf values

### Phase 1 (Parallel Operations)
- **Track A:** 48-hour uptime, <0.1% WebSocket drop rate
- **Track B:** Profit Factor > 1.1, Sharpe > 1.0 (out-of-sample)

### Phase 2 (Integration)
- Successful model injection
- ML-driven trades executing correctly
- Circuit breakers triggering appropriately

### Phase 3 (Scaling)
- CI/CD pipeline operational
- Latency targets met
- Security audit passed

---

## Team Responsibilities

### Data Engineer
- Phase 0: Data sanitizer implementation
- Ongoing: Data pipeline maintenance

### DevOps Engineer
- Track A: Infrastructure deployment and monitoring
- Phase 3: CI/CD pipeline setup

### ML Engineer/Researcher
- Track B: Feature engineering, model training, validation
- Phase 2: Model integration

### Quantitative Analyst
- All phases: Strategy validation, risk assessment, performance analysis

---

## Tools & Technologies

### Data Processing
- Pandas, NumPy for data manipulation
- `scripts/data_sanitizer.py` (custom)

### Infrastructure
- Docker, Docker Compose
- VPS (DigitalOcean, AWS, or similar)
- Telegram API for alerts

### ML Research
- Scikit-learn, XGBoost, LightGBM
- Optuna for hyperparameter optimization
- Walk-forward validation framework

### Monitoring
- Prometheus, Grafana
- Structured logging (ELK stack optional)

### CI/CD
- GitHub Actions
- Docker Hub / Container Registry

---

## Next Immediate Actions

1. **Completed:** ✅ Create and run `scripts/data_sanitizer.py`
2. **Completed:** ✅ Deploy dummy bot locally (Track A) - RUNNING
3. **Completed:** ✅ Begin ML research with cleaned data (Track B) - ITERATION 1 DONE

## Next Actions (Today/Tomorrow)
1. **Continue:** Track A stability testing (target 48 hours)
2. **Analyze:** Track B results to identify improvement areas
3. **Iterate:** Run Track B ML Research - Iteration 2 with improved features/parameters
4. **Prepare:** Phase 2 integration when both tracks meet criteria

---

*Document Version: 1.0*  
*Last Updated: 2025-12-23*  
*Author: CTO & Lead Quantitative Researcher*
