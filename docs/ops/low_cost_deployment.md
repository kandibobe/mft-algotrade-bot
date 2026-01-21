# Low-Cost VPS Deployment Guide

This guide describes how to run Stoic Citadel V7 on a budget (approx. €10-20/month) while maintaining institutional-grade reliability.

## 1. Recommended Hardware
- **Provider:** Hetzner Cloud, DigitalOcean, or Linode.
- **Plan:** Minimum 4 vCPU, 8GB RAM (e.g., Hetzner CPX21 or CX31).
- **OS:** Ubuntu 22.04 LTS.

## 2. Resource Optimization Strategies

### A. Off-load ML Training
Training models on a low-cost VPS is slow and may cause Out-Of-Memory (OOM) errors. 
- **Strategy:** Train models on your local machine (with GPU/high RAM).
- **Action:** Copy `.joblib` or `.onnx` files to `user_data/models/` on the VPS.
- **Config:** Set `optimize_hyperparams: false` and `feature_selection: false` in production config to prevent re-running heavy tasks.

### B. Efficient Database Management
PostgreSQL is great but can consume significant RAM.
- **Strategy:** Use the provided Docker-compose limits.
- **Action:** Ensure `deploy/docker-compose.yml` has `memory_limit: 1G` for Postgres.

### C. Docker Multi-Stage Build
Reduces image size from ~3GB to ~800MB.
- **Action:** Use the optimized `deploy/docker/Dockerfile`.

### D. Reduce Log Verbosity
Logs can fill up disk space quickly.
- **Action:** Set `log_level: "INFO"` in `unified_config.py` instead of `DEBUG`.
- **Retention:** Use the provided `scripts/maintenance/cleanup_docker.sh`.

## 3. Simplified Architecture (No-Redis Option)
If you are running a single bot instance, you can disable Redis to save ~200MB RAM.
- **Strategy:** Use in-memory `MockFeatureStore` with persistence to disk.
- **Action:** Set `use_redis: false` in `unified_config.py`.

## 4. Step-by-Step Installation

```bash
# 1. Update System
sudo apt update && sudo apt upgrade -y

# 2. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 3. Clone Repository
git clone https://github.com/kandibobe/mft-algotrade-bot.git
cd mft-algotrade-bot

# 4. Configure Environment
cp .env.example .env
nano .env # Add your API Keys

# 5. Start with Production Profile
docker-compose -f docker-compose.yml up -d
```

## 5. Monitoring on a Budget
Instead of a separate monitoring server, run Grafana and Prometheus on the same VPS using the `deploy/docker-compose.monitoring.yml` file, but increase swap space if needed.

```bash
# Add 4GB swap space to prevent OOM
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```