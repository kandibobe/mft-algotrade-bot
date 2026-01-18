# Stoic Citadel - Operational Runbook (Playbook)

**Version:** 1.1 (Final Polish)
**Status:** PROD READY
**Role:** System Operator / Quant Trader

---

## 1. Daily Routine (Morning Checks)

Perform these checks every morning (e.g., 08:00 UTC).

### 1.1 Health Check
1.  **Access Grafana:** `http://localhost:3001` (admin/admin).
2.  **Verify Heartbeat:** Check "System Uptime" panel.
3.  **Check Errors:** Ensure "Error Rate" is < 1%.

### 1.2 Risk Status
1.  **Check Logs:**
    ```bash
    tail -n 100 user_data/logs/stoic_citadel.log | grep "Circuit Breaker"
    ```
    *   Expected: No "Circuit Breaker TRIPPED" messages in the last 24h.
2.  **Verify System Integrity:**
    Run critical tests to ensure risk engine is active.
    ```bash
    pytest tests/test_risk/test_circuit_breaker.py
    ```
    *   Expected: All tests passed.
3.  **Verify HRP Weights:**
    ```bash
    tail -n 100 user_data/logs/stoic_citadel.log | grep "Updated HRP weights"
    ```
    *   Expected: At least one update in the last hour.

### 1.3 Portfolio Rebalancing
*   The system rebalances automatically via HRP.
*   **Manual Check:** Ensure `BTC/USDT` and `ETH/USDT` allocation matches the latest HRP log entry (±5%).

---

## 2. Emergency Procedures ("Kill Switch")

Use these procedures if the bot behaves erratically or the market crashes uncontrollably.

### Level 1: Stop Trading (Graceful)
Stops entering new trades but manages existing ones.
1.  **Command:**
    ```bash
    docker-compose exec freqtrade freqtrade stopentry
    ```
2.  **Verification:** Check Telegram for "Stopping buying" message.

### Level 2: Force Exit (Liquidate All)
Closes all open positions immediately at market price.
1.  **Command:**
    ```bash
    docker-compose exec freqtrade freqtrade forceexit all
    ```
2.  **Verification:** Watch logs for "Force exiting trade...".

### Level 3: Hard Shutdown (Process Kill)
Use only if the bot is stuck or sending erroneous orders.
1.  **Command:**
    ```bash
    docker-compose stop freqtrade
    ```
2.  **Note:** This leaves positions open on the exchange. You MUST log in to Binance manually to manage them.

---

## 3. Incident Management ("What to do if...")

### 3.1 Bot Crashed / Loop Stopped
*   **Symptom:** No updates in Telegram for > 15 mins. No log updates.
*   **Action:**
    1. Check Docker status: `docker-compose ps`
    2. Inspect logs for OOM or Fatal errors: `docker-compose logs --tail=100 freqtrade`
    3. Restart: `docker-compose restart freqtrade`

### 3.2 Exchange API Down / Connectivity Issues
*   **Symptom:** Logs show "ExchangeNotAvailable" or "RequestTimeout".
*   **Action:**
    1. Check exchange status page (e.g., Binance Status).
    2. If down, wait for recovery. The bot has automatic retries.
    3. If prolonged (> 1 hour), consider manual position management on the exchange website.

### 3.3 Funds Missing / Unexpected Balance Drop
*   **Symptom:** Portfolio value in Grafana or Exchange drops significantly without corresponding trades.
*   **Action:**
    1. **IMMEDIATE:** Lock API keys on the exchange (delete or disable trading).
    2. Run Reconciliation script:
       ```bash
       python scripts/ops/reconcile.py
       ```
    3. Compare `user_data/logs/trades.sqlite` with exchange trade history.
    4. Check for unauthorized logins on the exchange account.

### 3.4 Circuit Breaker Tripped
*   **Symptom:** Logs show "Circuit breaker open", no new trades.
*   **Action:**
    1. Analyze the loss cause (Flash crash? Strategy bug?).
    2. Check `user_data/circuit_breaker_state.json`.
    3. If safe, reset: Restart container or wait for daily reset (00:00 UTC).

---

## 4. Maintenance Procedures

### 4.1 Update Strategy/Code
1.  **Pull changes:**
    ```bash
    git pull
    ```
2.  **Rebuild Docker:**
    ```bash
    docker-compose up -d --build
    ```
3.  **Verify:** Check logs for successful startup.

### 4.2 Database Maintenance
If the database grows too large (>1GB):
1.  **Stop Bot:** `docker-compose stop`
2.  **Vacuum:**
    ```bash
    docker-compose up -d postgres
    docker-compose exec postgres psql -U stoic_trader -d trading_analytics -c "VACUUM FULL;"
    ```
3.  **Restart:** `docker-compose up -d`

---

## 5. Security Checklist (Pre-Money)

- [ ] **API Keys:** "Trade" enabled, "Withdraw" **DISABLED**.
- [ ] **IP Whitelisting:** API keys restricted to your server's IP.
- [ ] **2FA:** Enabled on Exchange and Telegram account.
- [ ] **Environment:** `.env` file is NOT in the repository.
- [ ] **Secrets:** All keys are encrypted via `src/utils/vault_client.py` if using production vault.

---

## 6. Contact & Support
*   **Lead Developer:** [Your Contact]
*   **Repository:** https://github.com/kandibobe/mft-algotrade-bot
