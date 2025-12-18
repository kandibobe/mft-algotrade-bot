# ==============================================================================
# Stoic Citadel - Algorithmic Trading Bot Makefile
# ==============================================================================
# Unified development workflow for algorithmic trading infrastructure
#
# Features:
#   - ML Pipeline: Triple Barrier Labeling, Feature Selection, Model Training
#   - Smart Order Execution: Limit order state machine with fee optimization
#   - Risk Management: Dynamic position sizing, Market regime filter
#   - Testing: 190+ unit tests with coverage
#
# Author: Stoic Citadel Team
# License: MIT
# ==============================================================================

.DEFAULT_GOAL := help
.PHONY: help setup start stop restart status logs test lint backtest research clean trade trade-dry trade-live download verify

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DOCKER_COMPOSE := docker-compose
DOCKER_COMPOSE_TEST := docker-compose -f docker-compose.test.yml
STRATEGY ?= StoicEnsembleStrategy
SERVICE ?= freqtrade
PYTEST_ARGS ?= -v --tb=short
TIMERANGE ?= 20240101-

# Colors
GREEN  := \033[0;32m
YELLOW := \033[1;33m
RED    := \033[0;31m
CYAN   := \033[0;36m
NC     := \033[0m

# ==============================================================================
# HELP
# ==============================================================================

help: ## Show this help message
	@echo ""
	@echo "$(CYAN)╔════════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(CYAN)║           STOIC CITADEL - ALGORITHMIC TRADING BOT                 ║$(NC)"
	@echo "$(CYAN)╚════════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)Development Workflow:$(NC)"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(CYAN)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Examples:$(NC)"
	@echo "  make setup                          # Initial setup"
	@echo "  make test                           # Run all tests"
	@echo "  make lint                           # Check code quality"
	@echo "  make train                          # Train ML model"
	@echo "  make optimize-ml                    # Hyperopt with Optuna"
	@echo "  make backtest STRATEGY=MyStrategy   # Run backtest"
	@echo "  make logs SERVICE=jupyter           # View Jupyter logs"
	@echo ""

# ==============================================================================
# SETUP & INITIALIZATION
# ==============================================================================

setup: ## Run interactive setup wizard
	@echo "$(CYAN)Running Stoic Citadel Setup Wizard...$(NC)"
	@python3 scripts/setup_wizard.py || ./scripts/citadel.sh setup
	@echo "$(GREEN)✅ Setup completed!$(NC)"

check-env: ## Check if .env file exists
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)⚠️  .env file not found. Creating from template...$(NC)"; \
		cp .env.example .env; \
		echo "$(GREEN)✅ Created .env file. Please configure it before proceeding.$(NC)"; \
	fi

# ==============================================================================
# DOCKER OPERATIONS
# ==============================================================================

##@ Docker Operations

start: check-env ## Start all services
	@echo "$(CYAN)Starting Stoic Citadel services...$(NC)"
	@$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)✅ All services started!$(NC)"
	@echo ""
	@echo "$(CYAN)Access Points:$(NC)"
	@echo "  📊 FreqUI Dashboard:  http://localhost:3000"
	@echo "  🔬 Jupyter Lab:       http://localhost:8888 (token: stoic2024)"
	@echo "  🐳 Portainer:         http://localhost:9000"
	@echo ""

stop: ## Stop all services
	@echo "$(YELLOW)Stopping all services...$(NC)"
	@$(DOCKER_COMPOSE) down
	@echo "$(GREEN)✅ All services stopped!$(NC)"

restart: ## Restart all services
	@echo "$(YELLOW)Restarting services...$(NC)"
	@$(MAKE) stop
	@sleep 2
	@$(MAKE) start

status: ## Show status of all services
	@echo "$(CYAN)Service Status:$(NC)"
	@$(DOCKER_COMPOSE) ps

logs: ## View service logs (default: freqtrade, use SERVICE=name)
	@echo "$(CYAN)Logs for $(SERVICE):$(NC)"
	@$(DOCKER_COMPOSE) logs -f --tail=100 $(SERVICE)

build: ## Rebuild all Docker containers
	@echo "$(CYAN)Building Docker containers...$(NC)"
	@$(DOCKER_COMPOSE) build --no-cache
	@echo "$(GREEN)✅ Build completed!$(NC)"

# ==============================================================================
# TESTING & QUALITY ASSURANCE
# ==============================================================================

##@ Testing & Quality

test: ## Run all tests
	@echo "$(CYAN)Running test suite...$(NC)"
	@$(DOCKER_COMPOSE_TEST) run --rm test pytest $(PYTEST_ARGS) tests/
	@echo "$(GREEN)✅ Tests completed!$(NC)"

test-unit: ## Run unit tests only
	@echo "$(CYAN)Running unit tests...$(NC)"
	@$(DOCKER_COMPOSE_TEST) run --rm test pytest $(PYTEST_ARGS) tests/test_strategies/
	@echo "$(GREEN)✅ Unit tests completed!$(NC)"

test-integration: ## Run integration tests
	@echo "$(CYAN)Running integration tests...$(NC)"
	@$(DOCKER_COMPOSE_TEST) run --rm test pytest $(PYTEST_ARGS) tests/test_integration/
	@echo "$(GREEN)✅ Integration tests completed!$(NC)"

test-coverage: ## Run tests with coverage report
	@echo "$(CYAN)Running tests with coverage...$(NC)"
	@$(DOCKER_COMPOSE_TEST) run --rm test pytest --cov=src --cov-report=html --cov-report=term $(PYTEST_ARGS) tests/
	@echo "$(GREEN)✅ Coverage report generated in htmlcov/$(NC)"

lint: ## Run all linters (flake8, black, mypy)
	@echo "$(CYAN)Running linters...$(NC)"
	@echo "$(YELLOW)▶ Checking code formatting with black...$(NC)"
	@docker run --rm -v $$(pwd):/app -w /app python:3.11-slim sh -c "pip install -q black && black --check --line-length 88 user_data/strategies/ scripts/" || (echo "$(RED)❌ Black formatting issues found. Run 'make format' to fix.$(NC)" && exit 1)
	@echo "$(YELLOW)▶ Checking code style with flake8...$(NC)"
	@docker run --rm -v $$(pwd):/app -w /app python:3.11-slim sh -c "pip install -q flake8 && flake8 --max-line-length=88 --extend-ignore=E203 user_data/strategies/ scripts/" || (echo "$(RED)❌ Flake8 issues found.$(NC)" && exit 1)
	@echo "$(YELLOW)▶ Checking types with mypy...$(NC)"
	@docker run --rm -v $$(pwd):/app -w /app python:3.11-slim sh -c "pip install -q mypy && mypy --ignore-missing-imports user_data/strategies/ scripts/" || (echo "$(RED)❌ MyPy type issues found.$(NC)" && exit 1)
	@echo "$(GREEN)✅ All linting checks passed!$(NC)"

format: ## Auto-format code with black
	@echo "$(CYAN)Formatting code with black...$(NC)"
	@docker run --rm -v $$(pwd):/app -w /app python:3.11-slim sh -c "pip install -q black && black --line-length 88 user_data/strategies/ scripts/"
	@echo "$(GREEN)✅ Code formatted!$(NC)"

pre-commit: ## Install and run pre-commit hooks
	@echo "$(CYAN)Running pre-commit hooks...$(NC)"
	@pip install -q pre-commit || echo "$(YELLOW)⚠️  Please install pre-commit: pip install pre-commit$(NC)"
	@pre-commit run --all-files || echo "$(YELLOW)⚠️  Some pre-commit checks failed$(NC)"

# ==============================================================================
# TRADING OPERATIONS
# ==============================================================================

##@ Trading Operations

trade-dry: check-env ## Start trading in dry-run mode (paper trading)
	@echo "$(CYAN)Starting trading bot in DRY-RUN mode...$(NC)"
	@$(DOCKER_COMPOSE) up -d freqtrade frequi
	@sleep 3
	@echo "$(GREEN)✅ Trading bot started (dry-run mode)!$(NC)"
	@echo ""
	@echo "$(CYAN)Monitor:$(NC)"
	@echo "  📊 Dashboard: http://localhost:3000"
	@echo "  📋 Logs:      make logs SERVICE=freqtrade"
	@echo ""

trade-live: check-env ## Start LIVE trading (USE WITH EXTREME CAUTION!)
	@echo "$(RED)╔═══════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(RED)║                    LIVE TRADING MODE                          ║$(NC)"
	@echo "$(RED)║                                                               ║$(NC)"
	@echo "$(RED)║  ⚠️  WARNING: THIS WILL USE REAL MONEY! ⚠️                     ║$(NC)"
	@echo "$(RED)║                                                               ║$(NC)"
	@echo "$(RED)║  Checklist:                                                   ║$(NC)"
	@echo "$(RED)║  [ ] Tested extensively in dry-run                            ║$(NC)"
	@echo "$(RED)║  [ ] API keys configured with trading permissions            ║$(NC)"
	@echo "$(RED)║  [ ] Risk limits set in config_production.json                ║$(NC)"
	@echo "$(RED)║  [ ] Telegram notifications enabled                           ║$(NC)"
	@echo "$(RED)║  [ ] Monitoring and alerts configured                         ║$(NC)"
	@echo "$(RED)╚═══════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@read -p "Type 'I UNDERSTAND THE RISKS' to proceed: " confirm; \
	if [ "$$confirm" != "I UNDERSTAND THE RISKS" ]; then \
		echo "$(YELLOW)⚠️  Live trading cancelled. Stay safe!$(NC)"; \
		exit 1; \
	fi
	@echo "$(RED)⚠️  REMEMBER TO SET dry_run: false IN config_production.json$(NC)"
	@$(DOCKER_COMPOSE) up -d freqtrade frequi
	@echo "$(GREEN)✅ Live trading started!$(NC)"
	@echo "$(RED)⚠️  Monitor closely! Check logs regularly!$(NC)"

backtest: check-env ## Run backtest (use STRATEGY=name, TIMERANGE=20240101-)
	@echo "$(CYAN)Running backtest for strategy: $(STRATEGY)$(NC)"
	@$(DOCKER_COMPOSE) run --rm freqtrade backtesting \
		--strategy $(STRATEGY) \
		--timerange $(TIMERANGE) \
		--enable-protections
	@echo "$(GREEN)✅ Backtest completed!$(NC)"

backtest-validate: check-env ## Run backtest with walk-forward validation
	@echo "$(CYAN)Running walk-forward validation...$(NC)"
	@$(DOCKER_COMPOSE) run --rm jupyter python /home/jovyan/scripts/walk_forward.py
	@echo "$(GREEN)✅ Validation completed!$(NC)"

hyperopt: check-env ## Run hyperparameter optimization
	@echo "$(CYAN)Running hyperparameter optimization for $(STRATEGY)...$(NC)"
	@$(DOCKER_COMPOSE) run --rm freqtrade hyperopt \
		--strategy $(STRATEGY) \
		--hyperopt-loss SharpeHyperOptLoss \
		--epochs 500 \
		--spaces buy sell
	@echo "$(GREEN)✅ Hyperopt completed!$(NC)"

# ==============================================================================
# ML PIPELINE
# ==============================================================================

##@ ML Pipeline

train: check-env ## Train ML model with proper data split
	@echo "$(CYAN)Training ML model...$(NC)"
	@$(DOCKER_COMPOSE) run --rm jupyter python -c "\
from src.ml.training import FeatureEngineer, TripleBarrierLabeler, TripleBarrierConfig, ModelTrainer, TrainingConfig; \
import pandas as pd; \
from sklearn.model_selection import train_test_split; \
df = pd.read_parquet('user_data/data/BTC_USDT-1h.parquet'); \
labeler = TripleBarrierLabeler(TripleBarrierConfig(take_profit_pct=0.02, stop_loss_pct=0.01)); \
df = labeler.create_labels(df); \
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False); \
engineer = FeatureEngineer(); \
train_features = engineer.fit_transform(train_df); \
test_features = engineer.transform(test_df); \
engineer.save_scaler('user_data/models/scaler.joblib'); \
print('Training model...'); \
trainer = ModelTrainer(TrainingConfig(model_type='lightgbm')); \
model, metrics = trainer.train(train_features[engineer.get_feature_names()].dropna(), train_features.loc[train_features[engineer.get_feature_names()].dropna().index, 'label']); \
print(f'Metrics: {metrics}')"
	@echo "$(GREEN)✅ Model trained!$(NC)"

optimize-ml: check-env ## Run ML hyperparameter optimization with Optuna
	@echo "$(CYAN)Running ML hyperparameter optimization...$(NC)"
	@$(DOCKER_COMPOSE) run --rm jupyter python scripts/optimize_strategy.py --n-trials 100
	@echo "$(GREEN)✅ Optimization completed!$(NC)"

feature-select: check-env ## Run feature selection (SHAP + Permutation Importance)
	@echo "$(CYAN)Running feature selection...$(NC)"
	@$(DOCKER_COMPOSE) run --rm jupyter python -c "\
from src.ml.training import FeatureSelector, FeatureSelectionConfig, FeatureEngineer; \
import pandas as pd; \
df = pd.read_parquet('user_data/data/BTC_USDT-1h.parquet'); \
engineer = FeatureEngineer(); \
features = engineer.fit_transform(df); \
X = features[engineer.get_feature_names()].dropna(); \
y = features.loc[X.index, 'close'].pct_change().shift(-1) > 0; \
selector = FeatureSelector(FeatureSelectionConfig(method='permutation')); \
X_selected, selected = selector.fit_transform(X, y.loc[X.index].dropna()); \
selector.save_selected_features(); \
print(selector.get_feature_report())"
	@echo "$(GREEN)✅ Feature selection completed!$(NC)"

# ==============================================================================
# RESEARCH & DATA
# ==============================================================================

##@ Research & Data

research: ## Start Jupyter Lab for strategy research
	@echo "$(CYAN)Starting Jupyter Lab...$(NC)"
	@$(DOCKER_COMPOSE) up -d jupyter
	@sleep 3
	@echo "$(GREEN)✅ Jupyter Lab is running!$(NC)"
	@echo ""
	@echo "$(CYAN)Access: http://localhost:8888$(NC)"
	@echo "$(CYAN)Token:  stoic2024$(NC)"
	@echo ""

download: check-env ## Download historical market data
	@echo "$(CYAN)Downloading historical data...$(NC)"
	@chmod +x ./scripts/download_data.sh
	@./scripts/download_data.sh 90 5m
	@echo "$(GREEN)✅ Data download completed!$(NC)"

verify: ## Verify data quality and integrity
	@echo "$(CYAN)Verifying data quality...$(NC)"
	@$(DOCKER_COMPOSE) run --rm jupyter python /home/jovyan/scripts/verify_data.py
	@echo "$(GREEN)✅ Data verification completed!$(NC)"

# ==============================================================================
# MONITORING & MAINTENANCE
# ==============================================================================

##@ Monitoring & Maintenance

monitoring: ## Start monitoring stack (Prometheus + Grafana)
	@echo "$(CYAN)Starting monitoring stack...$(NC)"
	@docker-compose -f docker-compose.monitoring.yml up -d
	@sleep 5
	@echo "$(GREEN)✅ Monitoring stack started!$(NC)"
	@echo ""
	@echo "$(CYAN)Access Points:$(NC)"
	@echo "  📈 Grafana:    http://localhost:3001 (admin/admin)"
	@echo "  📊 Prometheus: http://localhost:9090"
	@echo ""

monitoring-stop: ## Stop monitoring stack
	@echo "$(YELLOW)Stopping monitoring stack...$(NC)"
	@docker-compose -f docker-compose.monitoring.yml down
	@echo "$(GREEN)✅ Monitoring stopped!$(NC)"

clean: ## Remove containers and volumes (keeps data)
	@echo "$(YELLOW)⚠️  This will remove all containers and networks...$(NC)"
	@read -p "Are you sure? (yes/no): " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		echo "$(CYAN)Cleaning up...$(NC)"; \
		$(DOCKER_COMPOSE) down; \
		echo "$(GREEN)✅ Cleanup completed!$(NC)"; \
	else \
		echo "$(YELLOW)Cleanup cancelled.$(NC)"; \
	fi

clean-all: ## Remove EVERYTHING including volumes and data
	@echo "$(RED)⚠️  THIS WILL DELETE ALL DATA INCLUDING TRADE HISTORY!$(NC)"
	@read -p "Type 'DELETE EVERYTHING' to confirm: " confirm; \
	if [ "$$confirm" = "DELETE EVERYTHING" ]; then \
		echo "$(CYAN)Removing all containers, volumes, and data...$(NC)"; \
		$(DOCKER_COMPOSE) down -v; \
		docker-compose -f docker-compose.test.yml down -v 2>/dev/null || true; \
		docker-compose -f docker-compose.monitoring.yml down -v 2>/dev/null || true; \
		rm -rf user_data/data/* user_data/logs/* htmlcov/ .coverage .pytest_cache/ .mypy_cache/; \
		echo "$(GREEN)✅ Everything cleaned!$(NC)"; \
	else \
		echo "$(YELLOW)Cleanup cancelled.$(NC)"; \
	fi

# ==============================================================================
# DEPLOYMENT
# ==============================================================================

##@ Deployment

deploy: ## Deploy to production VPS
	@echo "$(CYAN)Deploying to production...$(NC)"
	@chmod +x ./scripts/deploy.sh
	@./scripts/deploy.sh
	@echo "$(GREEN)✅ Deployment script executed!$(NC)"

validate-config: ## Validate configuration files
	@echo "$(CYAN)Validating configuration...$(NC)"
	@python3 scripts/validate_config.py
	@echo "$(GREEN)✅ Configuration valid!$(NC)"

# ==============================================================================
# UTILITIES
# ==============================================================================

##@ Utilities

shell-freqtrade: ## Open shell in Freqtrade container
	@$(DOCKER_COMPOSE) exec freqtrade bash

shell-jupyter: ## Open shell in Jupyter container
	@$(DOCKER_COMPOSE) exec jupyter bash

list-strategies: ## List all available strategies
	@echo "$(CYAN)Available strategies:$(NC)"
	@$(DOCKER_COMPOSE) run --rm freqtrade list-strategies

list-pairs: ## List configured trading pairs
	@echo "$(CYAN)Configured trading pairs:$(NC)"
	@$(DOCKER_COMPOSE) run --rm freqtrade list-pairs

db-backup: ## Backup trading database
	@echo "$(CYAN)Backing up database...$(NC)"
	@mkdir -p backups
	@cp user_data/tradesv3.sqlite backups/tradesv3_$$(date +%Y%m%d_%H%M%S).sqlite
	@echo "$(GREEN)✅ Database backed up to backups/$(NC)"

.SILENT: help
