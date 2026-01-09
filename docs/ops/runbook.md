# Stoic Citadel - Operational Playbook

**Version:** 1.0 (Soft Launch)
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
2.  **Verify HRP Weights:**
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

## 3. Maintenance Procedures

### 3.1 Update Strategy/Code
1.  **Pull changes:**
    ```bash
    git pull
    ```
2.  **Rebuild Docker:**
    ```bash
    docker-compose up -d --build
    ```
3.  **Verify:** Check logs for successful startup.

### 3.2 Database Maintenance
If the database grows too large (>1GB):
1.  **Stop Bot:** `docker-compose stop`
2.  **Vacuum:**
    ```bash
    docker-compose up -d postgres
    docker-compose exec postgres psql -U stoic_trader -d trading_analytics -c "VACUUM FULL;"
    ```
3.  **Restart:** `docker-compose up -d`

---

## 4. Troubleshooting Common Issues

| Issue | Symptom | Fix |
| :--- | :--- | :--- |
| **Circuit Breaker Trip** | Logs show "Circuit breaker open", no new trades. | 1. Analyze loss cause.<br>2. If safe, reset: Restart container or wait for daily reset (00:00 UTC). |
| **HRP Weight Stale** | No "Updated HRP weights" log for >2 hours. | Check internet connection / exchange API status. Restart bot. |
| **Docker OOM** | Container crashes with exit code 137. | Increase memory limit in `docker-compose.yml`. |
| **Redis Auth Error** | Logs: "Authentication required". | Check `REDIS_PASSWORD` in `.env` matches `docker-compose.yml`. |

---

## 5. Contact & Support
*   **Lead Developer:** [Your Contact]
*   **Repository:** https://github.com/kandibobe/mft-algotrade-bot
