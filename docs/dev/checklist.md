# Definition of Done (DoD) & Quality Checklist

This checklist must be verified by Cline for every significant code change or feature implementation.

## 1. Core Logic & Safety
- [ ] **Risk Gate:** Does the change respect `src/risk/risk_manager.py`?
- [ ] **Unified Config:** Are all parameters retrieved from `src/config/unified_config.py`? (Zero hardcoded values).
- [ ] **Concurrency:** If modifying AsyncIO code, have potential race conditions been addressed?
- [ ] **Error Handling:** Does the code use `try/except` blocks that prevent system-wide crashes while logging enough context?

## 2. Architecture Compliance
- [ ] **Layer Boundaries:** Does the change violate the Macro/Micro layer separation?
- [ ] **Hybrid Bridge:** Are all cross-layer communications routed through `HybridConnector`?
- [ ] **No `user_data` Imports:** Does `src/` avoid importing from `user_data/`?

## 3. Data & ML
- [ ] **Vectorization:** Are DataFrame operations vectorized? (No `for` loops).
- [ ] **Feature Store:** Are new features registered in `src/ml/feature_store.py`?

## 4. Testing & Documentation
- [ ] **Tests:** Is there at least one new unit or integration test covering the change?
- [ ] **Logging:** Does the code use structured logging via `src/utils/logger.py`?
- [ ] **Docs:** Have relevant `.md` files in `docs/` been updated?

## 🧠 Critical Analysis Pass (Pre-Flight)
- [ ] **Scalability:** Will this work with 100x more data?
- [ ] **Latency:** Does this add blocking calls to an async loop?
- [ ] **Security:** Are there any exposed secrets or insecure API calls?