# Auto-Retrain Script for MLOps (Host Version)
# Usage: .\scripts\maintenance\auto_retrain.ps1

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

Write-Host "Starting Daily Retraining Pipeline..."

# 2. Run Feature Selection (Optional but Recommended)
Write-Host "Running feature selection..."
python scripts/analysis/select_features.py --pair BTC/USDT --months 3 --output user_data/config/features_btc.json
python scripts/analysis/select_features.py --pair ETH/USDT --months 3 --output user_data/config/features_eth.json

# 3. Train and Promote Models
Write-Host "Training and Promoting Production Models..."
python scripts/ml/retrain_production_model.py --pair BTC/USDT --params-file user_data/nightly_hyperopt/best_params_nightly.json
python scripts/ml/retrain_production_model.py --pair ETH/USDT --params-file user_data/nightly_hyperopt/best_params_nightly.json

# 4. (Optional) Trigger a system health check after retraining
# python scripts/maintenance/system_health_check.py

Write-Host "Retraining complete."