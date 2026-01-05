# 💼 Commercial Deployment Guide

This guide is designed for operators who manage **Stoic Citadel** instances for third-party clients or manage capital professionally. It covers infrastructure, security, onboarding, and reporting.

## 1. Deployment Tiers

Define clear service levels based on client capital and required complexity.

| Tier | Capital Range | Infrastructure | Strategy Complexity | Monitoring |
| :--- | :--- | :--- | :--- | :--- |
| **Standard** | < $50k | Shared VPS (Docker) | Single Strategy (Trend) | Daily Email |
| **Professional** | $50k - $200k | Dedicated Instance | Ensemble (Trend + Mean Rev) | Telegram Alerts |
| **Institutional** | > $200k | High-Availability Cluster | Full Ensemble + ML + HRP | Real-time Dashboard + Support |

---

## 2. Infrastructure Setup

### Recommended Specifications

**Professional / Institutional Node:**
- **CPU:** 4+ vCPUs (High Frequency preferred for MFT execution)
- **RAM:** 16GB - 32GB (Essential for ML model inference and backtesting)
- **Storage:** 100GB+ NVMe SSD (Fast I/O for database and logs)
- **Network:** Low latency to exchange servers (AWS Tokyo for Binance, etc.)

### Security Checklist (CRITICAL)

1.  **Firewall (UFW):**
    - Block all incoming ports except SSH (custom port), 80/443 (if using SSL).
    - Whitelist your static IP for SSH access.
2.  **VPN Access:**
    - Do NOT expose the FreqUI dashboard (port 8080) to the public internet.
    - Use WireGuard or OpenVPN to access the dashboard securely.
3.  **API Keys:**
    - **NEVER** enable "Withdrawal" permissions.
    - IP restrict keys to the server's static IP.
    - Use `secrets` management (Docker secrets or encrypted .env).
4.  **Database:**
    - Do not expose PostgreSQL port (5432) externally. Use internal Docker network only.

---

## 3. Client Onboarding Checklist

Follow this process for every new client deployment to ensure consistency and safety.

### Phase 1: Preparation
- [ ] Receive API Keys (Read + Trade ONLY) from client.
- [ ] Receive Telegram User ID for notifications.
- [ ] Determine risk profile (Conservative/Moderate/Aggressive).

### Phase 2: Configuration
- [ ] Clone repository to client's reserved infrastructure.
- [ ] Create `user_data/config/config_clientname.json` extending `config_production.json`.
- [ ] Set `stake_amount` and `max_open_trades` based on capital.
- [ ] Configure `stoploss` and `trailing_stop` based on risk profile.
- [ ] **Verify API connection:** Run dry-run for 1 hour.

### Phase 3: Launch
- [ ] Initialize database: `docker-compose run --rm freqtrade new-config --config ...`
- [ ] Start system: `docker-compose up -d`
- [ ] Verify logs: `docker-compose logs -f freqtrade`
- [ ] Send "System Online" notification to client via Telegram.

---

## 4. Maintenance & Operations

### Weekly Routines
- **Log Review:** Check `user_data/logs/freqtrade.log` for warnings/errors.
- **Database Cleanup:** Prune old trade data to maintain performance.
    ```bash
    docker-compose run --rm freqtrade db-cleanup --days 30
    ```
- **Performance Check:** Compare actual vs. backtest performance.

### Monthly Routines
- **Model Retraining:** Retrain ML models with the latest month's data.
    ```bash
    python src/ml/training/train.py --retrain --days 30
    ```
- **System Updates:** Pull latest git changes and update Docker images.
    ```bash
    git pull && docker-compose pull && docker-compose up -d
    ```

---

## 5. Reporting & Client Communication

### Automated Reporting
Use the built-in reporting tools to generate PDF/HTML reports for clients.

```bash
# Generate monthly performance report
python scripts/generate_report.py --user client_id --period 1M --format pdf
```

### Key Metrics to Report
1.  **ROI (Return on Investment):** Net profit %.
2.  **Max Drawdown:** The deepest decline from peak equity.
3.  **Win Rate:** Percentage of profitable trades.
4.  **Sharpe Ratio:** Risk-adjusted return metric.
5.  **Alpha:** Performance relative to Bitcoin (Buy & Hold).

---

## 6. Legal & Compliance Disclaimer

**IMPORTANT:**
- **Licensing:** Managing third-party capital usually requires a financial license (e.g., RIA in US, CySEC in EU).
- **Contracts:** Always have a signed Risk Disclosure and Management Agreement.
- **Liability:** Explicitly disclaim liability for software bugs or exchange failures.

*This guide is for technical deployment only and does not constitute legal advice.*
