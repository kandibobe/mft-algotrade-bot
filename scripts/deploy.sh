#!/bin/bash
# ==============================================================================
# Stoic Citadel - Production Deployment Script
# ==============================================================================
# Automated deployment for high-frequency trading infrastructure
#
# Usage:
#   ./scripts/deploy.sh [--setup|--data|--research|--validate]
#
# Commands:
#   --setup      Complete initial setup
#   --data       Download 2 years of historical data
#   --research   Launch research environment
#   --validate   Run walk-forward validation
# ==============================================================================

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="stoic_citadel"
DATA_PAIRS=("BTC/USDT" "ETH/USDT" "BNB/USDT" "SOL/USDT")
DATA_DAYS=730  # 2 years
TIMEFRAME="5m"
EXCHANGE="binance"

# ==============================================================================
# Helper Functions
# ==============================================================================

print_banner() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                                                                    ║${NC}"
    echo -e "${CYAN}║                    ${MAGENTA}STOIC CITADEL${CYAN}                                 ║${NC}"
    echo -e "${CYAN}║                                                                    ║${NC}"
    echo -e "${CYAN}║         ${YELLOW}Production HFT-Capable Trading Infrastructure${CYAN}       ║${NC}"
    echo -e "${CYAN}║                                                                    ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_step() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}► $1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        echo "Install Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        echo "Install Docker Compose: https://docs.docker.com/compose/install/"
        exit 1
    fi

    print_success "Docker and Docker Compose are installed"
}

check_permissions() {
    if ! docker ps &> /dev/null; then
        print_error "Cannot connect to Docker daemon"
        echo "Make sure Docker is running and you have permissions"
        echo "Try: sudo usermod -aG docker $USER"
        exit 1
    fi

    print_success "Docker permissions OK"
}

# ==============================================================================
# Setup Functions
# ==============================================================================

setup_directories() {
    print_step "Step 1: Creating Directory Structure"

    mkdir -p user_data/{config,strategies,data,logs,notebooks,hyperopt_results}
    mkdir -p research
    mkdir -p scripts
    mkdir -p docker

    print_success "Directories created"
}

setup_permissions() {
    print_step "Step 2: Setting File Permissions"

    chmod +x scripts/*.sh 2>/dev/null || true
    chmod +x scripts/*.py 2>/dev/null || true

    print_success "Permissions set"
}

pull_docker_images() {
    print_step "Step 3: Pulling Docker Images"

    print_info "Pulling Freqtrade image..."
    docker pull freqtradeorg/freqtrade:2024.11

    print_info "Pulling FreqUI image..."
    docker pull freqtradeorg/frequi:latest

    print_success "Docker images pulled"
}

build_research_lab() {
    print_step "Step 4: Building Research Lab Container"

    docker-compose build jupyter

    print_success "Research Lab built"
}

# ==============================================================================
# Data Functions
# ==============================================================================

download_data() {
    print_step "Downloading Historical Data"

    print_info "Exchange: $EXCHANGE"
    print_info "Timeframe: $TIMEFRAME"
    print_info "Days: $DATA_DAYS (2 years)"
    print_info "Pairs: ${DATA_PAIRS[*]}"

    echo ""

    for pair in "${DATA_PAIRS[@]}"; do
        print_info "Downloading $pair..."

        docker-compose run --rm freqtrade download-data \
            --exchange $EXCHANGE \
            --pairs "$pair" \
            --timeframe $TIMEFRAME \
            --days $DATA_DAYS \
            --data-format-ohlcv json

        print_success "$pair downloaded"
    done

    # Also download BTC/USDT on 1d for regime filter
    print_info "Downloading BTC/USDT 1d (regime filter)..."
    docker-compose run --rm freqtrade download-data \
        --exchange $EXCHANGE \
        --pairs "BTC/USDT" \
        --timeframe "1d" \
        --days $DATA_DAYS \
        --data-format-ohlcv json

    print_success "All data downloaded"

    # Verify data
    print_info "Verifying data quality..."
    if [ -f scripts/verify_data.py ]; then
        python3 scripts/verify_data.py
    else
        print_warning "Data verification script not found, skipping"
    fi
}

# ==============================================================================
# Research Functions
# ==============================================================================

launch_research_lab() {
    print_step "Launching Research Lab"

    docker-compose up -d jupyter

    sleep 3

    print_success "Jupyter Lab is running"
    echo ""
    print_info "Access: http://127.0.0.1:8888"
    print_info "Token: stoic2024"
    echo ""
    print_warning "SECURITY: Jupyter is bound to localhost only (127.0.0.1)"
    print_warning "To access from remote host, use SSH tunnel:"
    echo "  ssh -L 8888:localhost:8888 user@your-server"
    echo ""
}

# ==============================================================================
# Validation Functions
# ==============================================================================

run_walk_forward() {
    print_step "Running Walk-Forward Validation"

    print_info "This will validate the strategy on out-of-sample data"
    print_info "to prevent overfitting"
    echo ""

    read -p "Strategy name [StoicStrategyV1]: " strategy
    strategy=${strategy:-StoicStrategyV1}

    read -p "Train months [3]: " train_months
    train_months=${train_months:-3}

    read -p "Test months [1]: " test_months
    test_months=${test_months:-1}

    read -p "HyperOpt epochs [100]: " epochs
    epochs=${epochs:-100}

    echo ""
    print_info "Running walk-forward optimization..."
    print_warning "This may take several hours depending on data size"
    echo ""

    python3 scripts/walk_forward.py \
        --strategy "$strategy" \
        --config user_data/config/config_production.json \
        --train-months "$train_months" \
        --test-months "$test_months" \
        --epochs "$epochs" \
        --start-date 20220101 \
        --end-date 20241231

    if [ $? -eq 0 ]; then
        print_success "Walk-forward validation PASSED"
        echo ""
        print_info "Strategy is approved for paper trading"
        print_info "Next step: ./scripts/citadel.sh trade"
    else
        print_error "Walk-forward validation FAILED"
        echo ""
        print_warning "Strategy shows signs of overfitting"
        print_warning "Review the results and refine your strategy"
    fi
}

# ==============================================================================
# Complete Setup
# ==============================================================================

complete_setup() {
    print_banner

    print_info "This will set up the complete Stoic Citadel infrastructure"
    echo ""

    # Prerequisites check
    check_docker
    check_permissions

    # Setup steps
    setup_directories
    setup_permissions
    pull_docker_images
    build_research_lab

    print_success "Setup completed successfully!"
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    print_info "Next Steps:"
    echo "  1. Download data:    ./scripts/deploy.sh --data"
    echo "  2. Launch research:  ./scripts/deploy.sh --research"
    echo "  3. Develop strategy: Open Jupyter at http://127.0.0.1:8888"
    echo "  4. Validate:         ./scripts/deploy.sh --validate"
    echo "  5. Start trading:    ./scripts/citadel.sh trade"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# ==============================================================================
# Main Command Dispatcher
# ==============================================================================

case "${1:-}" in
    --setup)
        complete_setup
        ;;

    --data)
        print_banner
        download_data
        ;;

    --research)
        print_banner
        launch_research_lab
        ;;

    --validate)
        print_banner
        run_walk_forward
        ;;

    --help|-h|"")
        print_banner
        echo "Usage: ./scripts/deploy.sh [COMMAND]"
        echo ""
        echo "Commands:"
        echo "  --setup      Complete initial setup (directories, images, build)"
        echo "  --data       Download 2 years of historical data"
        echo "  --research   Launch Jupyter Lab research environment"
        echo "  --validate   Run walk-forward validation on strategy"
        echo ""
        echo "Typical workflow:"
        echo "  1. ./scripts/deploy.sh --setup"
        echo "  2. ./scripts/deploy.sh --data"
        echo "  3. ./scripts/deploy.sh --research"
        echo "  4. [Develop strategy in Jupyter]"
        echo "  5. ./scripts/deploy.sh --validate"
        echo "  6. ./scripts/citadel.sh trade  # Paper trading"
        echo ""
        ;;

    *)
        print_error "Unknown command: $1"
        echo "Run './scripts/deploy.sh --help' for usage"
        exit 1
        ;;
esac
