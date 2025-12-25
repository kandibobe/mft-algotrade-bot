# ==============================================================================
# Stoic Citadel - Professional Algorithmic Trading System
# ==============================================================================
# Production-ready trading infrastructure with ML pipeline and risk management
#
# Features:
#   - ML Pipeline: Triple Barrier Labeling, Feature Engineering, Model Training
#   - Smart Order Execution: Limit order state machine with fee optimization
#   - Risk Management: Dynamic position sizing, Market regime filter
#   - Testing: 190+ unit tests with coverage
#   - Monitoring: Health checks, metrics, structured logging
#
# Author: Stoic Citadel Team
# License: MIT
# ==============================================================================

.DEFAULT_GOAL := help
.PHONY: help setup install install-dev test lint format type-check clean backtest train trade monitor security

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DOCKER_COMPOSE := docker-compose
DOCKER_COMPOSE_TEST := docker-compose -f docker-compose.test.yml
STRATEGY ?= StoicEnsembleStrategyV4
SERVICE ?= freqtrade
PYTEST_ARGS ?= -v --tb=short
TIMERANGE ?= 20240101-

# Python configuration
PYTHON := python
PIP := pip
VENV := .venv
PYTHON_VERSION := 3.11

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
	@echo "$(CYAN)║           STOIC CITADEL - PROFESSIONAL TRADING SYSTEM             ║$(NC)"
	@echo "$(CYAN)╚════════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)Development Workflow:$(NC)"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(CYAN)%-25s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Examples:$(NC)"
	@echo "  make install                    # Install dependencies"
	@echo "  make install-dev                # Install development dependencies"
	@echo "  make test                       # Run all tests"
	@echo "  make lint                       # Check code quality"
	@echo "  make format                     # Auto-format code"
	@echo "  make security                   # Run security scans"
	@echo "  make train                      # Train ML model"
	@echo "  make backtest                   # Run backtest"
	@echo "  make monitor                    # Start monitoring stack"
	@echo "  make docker-up                  # Start Docker services"
	@echo ""

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

##@ Environment Setup

setup: ## Setup development environment (virtualenv + dependencies)
	@echo "$(CYAN)Setting up development environment...$(NC)"
	@$(PYTHON) -m venv $(VENV) || echo "$(YELLOW)Virtual environment already exists$(NC)"
	@echo "$(GREEN)✅ Virtual environment created$(NC)"
	@$(MAKE) install-dev
	@echo "$(GREEN)✅ Development environment ready$(NC)"

install: ## Install production dependencies
	@echo "$(CYAN)Installing production dependencies...$(NC)"
	@$(VENV)/Scripts/activate && $(PIP) install --upgrade pip
	@$(VENV)/Scripts/activate && $(PIP) install -e .
	@echo "$(GREEN)✅ Production dependencies installed$(NC)"

install-dev: ## Install all development dependencies
	@echo "$(CYAN)Installing development dependencies...$(NC)"
	@$(VENV)/Scripts/activate && $(PIP) install --upgrade pip
	@$(VENV)/Scripts/activate && $(PIP) install -e ".[dev]"
	@echo "$(GREEN)✅ Development dependencies installed$(NC)"

check-env: ## Check if .env file exists
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)⚠️  .env file not found. Creating from template...$(NC)"; \
		cp .env.example .env; \
		echo "$(GREEN)✅ Created .env file. Please configure it before proceeding.$(NC)"; \
	fi

# ==============================================================================
# CODE QUALITY
# ==============================================================================

##@ Code Quality

lint: ## Run all linters (ruff, black, isort, mypy)
	@echo "$(CYAN)Running code quality checks...$(NC)"
	@$(VENV)/Scripts/activate && ruff check src/ tests/ scripts/ || (echo "$(RED)❌ Ruff linting issues found$(NC)" && exit 1)
	@$(VENV)/Scripts/activate && black --check --line-length 100 src/ tests/ scripts/ || (echo "$(RED)❌ Black formatting issues found$(NC)" && exit 1)
	@$(VENV)/Scripts/activate && isort --check-only src/ tests/ scripts/ || (echo "$(RED)❌ Import sorting issues found$(NC)" && exit 1)
	@$(VENV)/Scripts/activate && mypy src/ --ignore-missing-imports || (echo "$(RED)❌ Type checking issues found$(NC)" && exit 1)
	@echo "$(GREEN)✅ All code quality checks passed!$(NC)"

format: ## Auto-format code with ruff, black and isort
	@echo "$(CYAN)Formatting code...$(NC)"
	@$(VENV)/Scripts/activate && ruff check --fix src/ tests/ scripts/
	@$(VENV)/Scripts/activate && black --line-length 100 src/ tests/ scripts/
	@$(VENV)/Scripts/activate && isort src/ tests/ scripts/
	@echo "$(GREEN)✅ Code formatted!$(NC)"

ruff: ## Run Ruff linter only
	@echo "$(CYAN)Running Ruff linter...$(NC)"
	@$(VENV)/Scripts/activate && ruff check src/ tests/ scripts/
	@echo "$(GREEN)✅ Ruff checks passed!$(NC)"

ruff-fix: ## Auto-fix Ruff issues
	@echo "$(CYAN)Fixing Ruff issues...$(NC)"
	@$(VENV)/Scripts/activate && ruff check --fix src/ tests/ scripts/
	@echo "$(GREEN)✅ Ruff fixes applied!$(NC)"

type-check: ## Run type checking with mypy
	@echo "$(CYAN)Running type checking...$(NC)"
	@$(VENV)/Scripts/activate && mypy src/ --ignore-missing-imports
	@echo "$(GREEN)✅ Type checking completed!$(NC)"

pre-commit: ## Install and run pre-commit hooks
	@echo "$(CYAN)Running pre-commit hooks...$(NC)"
	@$(VENV)/Scripts/activate && pre-commit install
	@$(VENV)/Scripts/activate && pre-commit run --all-files
	@echo "$(GREEN)✅ Pre-commit checks passed!$(NC)"

# ==============================================================================
# SECURITY
# ==============================================================================

##@ Security

security: ## Run security scans (bandit, safety)
	@echo "$(CYAN)Running security scans...$(NC)"
	@$(VENV)/Scripts/activate && bandit -r src/ -c pyproject.toml || (echo "$(YELLOW)⚠️  Bandit found potential security issues$(NC)" && exit 0)
	@$(VENV)/Scripts/activate && safety check || (echo "$(YELLOW)⚠️  Safety found vulnerable dependencies$(NC)" && exit 0)
	@echo "$(GREEN)✅ Security scans completed!$(NC)"

bandit: ## Run Bandit security scanner
	@echo "$(CYAN)Running Bandit security scan...$(NC)"
	@$(VENV)/Scripts/activate && bandit -r src/ -c pyproject.toml
	@echo "$(GREEN)✅ Bandit scan completed!$(NC)"

safety: ## Run Safety dependency checker
	@echo "$(CYAN)Running Safety dependency check...$(NC)"
	@$(VENV)/Scripts/activate && safety check
	@echo "$(GREEN)✅ Safety check completed!$(NC)"

# ==============================================================================
# TESTING
# ==============================================================================

##@ Testing

test: ## Run all tests
	@echo "$(CYAN)Running test suite...$(NC)"
	@$(VENV)/Scripts/activate && pytest tests/ $(PYTEST_ARGS)
	@echo "$(GREEN)✅ All tests passed!$(NC)"

test-unit: ## Run unit tests only
	@echo "$(CYAN)Running unit tests...$(NC)"
	@$(VENV)/Scripts/activate && pytest tests/test_unit/ $(PYTEST_ARGS)
	@echo "$(GREEN)✅ Unit tests completed!$(NC)"

test-integration: ## Run integration tests
	@echo "$(CYAN)Running integration tests...$(NC)"
	@$(VENV)/Scripts/activate && pytest tests/test_integration/ $(PYTEST_ARGS)
	@echo "$(GREEN)✅ Integration tests completed!$(NC)"

test-ml: ## Run ML pipeline tests
	@echo "$(CYAN)Running ML pipeline tests...$(NC)"
	@$(VENV)/Scripts/activate && pytest tests/test_ml/ $(PYTEST_ARGS)
	@echo "$(GREEN)✅ ML tests completed!$(NC)"

coverage: ## Run tests with coverage report
	@echo "$(CYAN)Running tests with coverage...$(NC)"
	@$(VENV)/Scripts/activate && pytest --cov=src --cov-report=html --cov-report=term $(PYTEST_ARGS) tests/
	@echo "$(GREEN)✅ Coverage report generated in htmlcov/$(NC)"

# ==============================================================================
# ML PIPELINE
# ==============================================================================

##@ ML Pipeline

train: ## Train ML model with proper data split
	@echo "$(CYAN)Training ML model...$(NC)"
	@$(VENV)/Scripts/activate && python scripts/train_models.py
	@echo "$(GREEN)✅ Model training completed!$(NC)"

optimize: ## Run hyperparameter optimization
	@echo "$(CYAN)Running hyperparameter optimization...$(NC)"
	@$(VENV)/Scripts/activate && python scripts/optimize_strategy.py --n-trials 100
	@echo "$(GREEN)✅ Hyperparameter optimization completed!$(NC)"

walk-forward: ## Run walk-forward validation
	@echo "$(CYAN)Running walk-forward validation...$(NC)"
	@$(VENV)/Scripts/activate && python scripts/walk_forward_analysis.py
	@echo "$(GREEN)✅ Walk-forward validation completed!$(NC)"

preflight: ## Run preflight checks
	@echo "$(CYAN)Running preflight checks...$(NC)"
	@$(VENV)/Scripts/activate && python scripts/preflight_check.py
	@echo "$(GREEN)✅ Preflight checks passed!$(NC)"

# ==============================================================================
# TRADING OPERATIONS
# ==============================================================================

##@ Trading Operations

backtest: ## Run backtest with default strategy
	@echo "$(CYAN)Running backtest for strategy: $(STRATEGY)$(NC)"
	@$(VENV)/Scripts/activate && freqtrade backtesting \
		--config user_data/config/config_backtest.json \
		--strategy $(STRATEGY) \
		--timerange $(TIMERANGE) \
		--enable-protections
	@echo "$(GREEN)✅ Backtest completed!$(NC)"

backtest-batch: ## Run batch backtesting
	@echo "$(CYAN)Running batch backtesting...$(NC)"
	@$(VENV)/Scripts/activate && python scripts/batch_backtest.py
	@echo "$(GREEN)✅ Batch backtesting completed!$(NC)"

trade-dry: ## Start trading in dry-run mode (paper trading)
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

# ==============================================================================
# DOCKER OPERATIONS
# ==============================================================================

##@ Docker Operations

docker-up: ## Start all Docker services
	@echo "$(CYAN)Starting Docker services...$(NC)"
	@$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)✅ All services started!$(NC)"
	@echo ""
	@echo "$(CYAN)Access Points:$(NC)"
	@echo "  📊 FreqUI Dashboard:  http://localhost:3000"
	@echo "  🔬 Jupyter Lab:       http://localhost:8888"
	@echo "  📈 Grafana:           http://localhost:3001"
	@echo ""

docker-stop: ## Stop all Docker services
	@echo "$(YELLOW)Stopping all services...$(NC)"
	@$(DOCKER_COMPOSE) down
	@echo "$(GREEN)✅ All services stopped!$(NC)"

docker-restart: ## Restart all Docker services
	@$(MAKE) docker-stop
	@sleep 2
	@$(MAKE) docker-up

docker-logs: ## View service logs
	@echo "$(CYAN)Logs for $(SERVICE):$(NC)"
	@$(DOCKER_COMPOSE) logs -f --tail=100 $(SERVICE)

docker-status: ## Show status of all services
	@echo "$(CYAN)Service Status:$(NC)"
	@$(DOCKER_COMPOSE) ps

# ==============================================================================
# MONITORING
# ==============================================================================

##@ Monitoring

monitor: ## Start monitoring stack (Prometheus + Grafana)
	@echo "$(CYAN)Starting monitoring stack...$(NC)"
	@docker-compose -f docker-compose.monitoring.yml up -d
	@sleep 5
	@echo "$(GREEN)✅ Monitoring stack started!$(NC)"
	@echo ""
	@echo "$(CYAN)Access Points:$(NC)"
	@echo "  📈 Grafana:    http://localhost:3001 (admin/admin)"
	@echo "  📊 Prometheus: http://localhost:9090"
	@echo ""

monitor-stop: ## Stop monitoring stack
	@echo "$(YELLOW)Stopping monitoring stack...$(NC)"
	@docker-compose -f docker-compose.monitoring.yml down
	@echo "$(GREEN)✅ Monitoring stopped!$(NC)"

health-check: ## Run health checks
	@echo "$(CYAN)Running health checks...$(NC)"
	@$(VENV)/Scripts/activate && python scripts/health_check.py
	@echo "$(GREEN)✅ Health checks completed!$(NC)"

# ==============================================================================
# CLEANUP
# ==============================================================================

##@ Cleanup

clean: ## Remove Python cache files and build artifacts
	@echo "$(CYAN)Cleaning up...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*.pyd" -delete
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".hypothesis" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✅ Cleanup completed!$(NC)"

clean-docker: ## Remove Docker containers and volumes (keeps data)
	@echo "$(YELLOW)⚠️  This will remove all containers and networks...$(NC)"
	@read -p "Are you sure? (yes/no): " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		echo "$(CYAN)Cleaning up Docker...$(NC)"; \
		$(DOCKER_COMPOSE) down; \
		echo "$(GREEN)✅ Docker cleanup completed!$(NC)"; \
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
		echo "$(GREEN)✅ Complete cleanup finished!$(NC)"; \
	fi
