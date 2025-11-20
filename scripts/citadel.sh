#!/bin/bash
# ==============================================================================
# Stoic Citadel - Master Control Script
# ==============================================================================
# Unified interface for managing the entire trading infrastructure
#
# Usage:
#   ./scripts/citadel.sh [command]
#
# Commands:
#   setup       - First-time setup (build containers)
#   start       - Start all services
#   stop        - Stop all services
#   restart     - Restart all services
#   logs        - View logs
#   status      - Show status of all services
#   research    - Open Jupyter Lab
#   trade       - Start trading bot (dry-run)
#   trade-live  - Start trading bot (LIVE - USE WITH CAUTION!)
#   backtest    - Run backtest
#   download    - Download historical data
#   verify      - Verify data quality
#   clean       - Clean up containers and volumes
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner
print_banner() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                                                                    ║${NC}"
    echo -e "${CYAN}║                    ${MAGENTA}STOIC CITADEL${CYAN}                                 ║${NC}"
    echo -e "${CYAN}║                                                                    ║${NC}"
    echo -e "${CYAN}║         ${YELLOW}Professional HFT-lite Trading Infrastructure${CYAN}            ║${NC}"
    echo -e "${CYAN}║                                                                    ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Helper functions
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

# Command functions
cmd_setup() {
    print_banner
    echo "Setting up Stoic Citadel for the first time..."
    echo ""

    print_info "Building Docker containers..."
    docker-compose build

    print_info "Creating necessary directories..."
    mkdir -p user_data/{config,strategies,data,notebooks,logs}
    mkdir -p research
    mkdir -p scripts

    print_success "Setup completed!"
    echo ""
    print_info "Next steps:"
    echo "  1. Configure Telegram (optional): edit user_data/config/config_production.json"
    echo "  2. Add exchange API keys: edit user_data/config/config_production.json"
    echo "  3. Download data: ./scripts/citadel.sh download"
    echo "  4. Start research: ./scripts/citadel.sh research"
    echo ""
}

cmd_start() {
    print_banner
    print_info "Starting all Stoic Citadel services..."
    docker-compose up -d
    echo ""
    print_success "All services started!"
    echo ""
    print_info "Access points:"
    echo "  📊 FreqUI Dashboard:  http://localhost:3000"
    echo "  🔬 Jupyter Lab:       http://localhost:8888 (token: stoic2024)"
    echo "  🐳 Portainer:         http://localhost:9000"
    echo ""
}

cmd_stop() {
    print_banner
    print_info "Stopping all services..."
    docker-compose down
    print_success "All services stopped!"
}

cmd_restart() {
    cmd_stop
    sleep 2
    cmd_start
}

cmd_logs() {
    SERVICE=${1:-freqtrade}
    print_banner
    print_info "Showing logs for: $SERVICE"
    echo ""
    docker-compose logs -f --tail=100 $SERVICE
}

cmd_status() {
    print_banner
    print_info "Service Status:"
    echo ""
    docker-compose ps
    echo ""
}

cmd_research() {
    print_banner
    print_info "Starting Jupyter Lab..."
    docker-compose up -d jupyter
    sleep 3
    echo ""
    print_success "Jupyter Lab is running!"
    echo ""
    print_info "Access: http://localhost:8888"
    print_info "Token: stoic2024"
    echo ""
    print_warning "Don't forget to stop it when done: ./scripts/citadel.sh stop"
    echo ""
}

cmd_trade() {
    print_banner
    print_warning "Starting trading bot in DRY-RUN mode (fake money)"
    echo ""
    docker-compose up -d freqtrade frequi
    sleep 3
    echo ""
    print_success "Trading bot started!"
    echo ""
    print_info "Monitor:"
    echo "  📊 Dashboard: http://localhost:3000"
    echo "  📋 Logs:      ./scripts/citadel.sh logs freqtrade"
    echo ""
}

cmd_trade_live() {
    print_banner
    print_error "╔═══════════════════════════════════════════════════════════════╗"
    print_error "║                    LIVE TRADING MODE                          ║"
    print_error "║                                                               ║"
    print_error "║  ⚠️  WARNING: THIS WILL USE REAL MONEY! ⚠️                     ║"
    print_error "║                                                               ║"
    print_error "║  Make sure you have:                                          ║"
    print_error "║  1. Tested extensively in dry-run                             ║"
    print_error "║  2. Set proper API keys with trading permissions              ║"
    print_error "║  3. Configured risk limits in config_production.json          ║"
    print_error "║  4. Enabled Telegram notifications                            ║"
    print_error "║  5. Set up monitoring and alerts                              ║"
    print_error "╚═══════════════════════════════════════════════════════════════╝"
    echo ""

    read -p "Type 'I UNDERSTAND THE RISKS' to proceed: " confirmation

    if [ "$confirmation" != "I UNDERSTAND THE RISKS" ]; then
        print_warning "Live trading cancelled. Stay safe!"
        exit 0
    fi

    print_warning "Switching to LIVE mode..."

    # Update config to set dry_run = false
    # (In production, you'd use a separate config file)
    print_error "⚠️  REMEMBER TO SET dry_run: false IN config_production.json"

    docker-compose up -d freqtrade frequi

    echo ""
    print_success "Live trading started!"
    print_warning "Monitor closely! Check logs regularly!"
    echo ""
}

cmd_backtest() {
    STRATEGY=${1:-StoicEnsembleStrategy}
    print_banner
    print_info "Running backtest for strategy: $STRATEGY"
    echo ""

    docker-compose run --rm freqtrade backtesting \
        --strategy $STRATEGY \
        --timerange 20240101-

    echo ""
    print_success "Backtest completed!"
}

cmd_download() {
    print_banner
    print_info "Downloading historical data..."
    echo ""

    # Make script executable
    chmod +x ./scripts/download_data.sh

    # Run download script
    ./scripts/download_data.sh 90 5m
}

cmd_verify() {
    print_banner
    print_info "Verifying data quality..."
    echo ""

    docker-compose run --rm jupyter python /home/jovyan/scripts/verify_data.py
}

cmd_clean() {
    print_banner
    print_warning "This will remove all containers and volumes!"
    read -p "Are you sure? (yes/no): " confirm

    if [ "$confirm" == "yes" ]; then
        print_info "Cleaning up..."
        docker-compose down -v
        print_success "Cleanup completed!"
    else
        print_info "Cleanup cancelled"
    fi
}

cmd_help() {
    print_banner
    echo "Available commands:"
    echo ""
    echo "  ${GREEN}setup${NC}       - First-time setup (build containers)"
    echo "  ${GREEN}start${NC}       - Start all services"
    echo "  ${GREEN}stop${NC}        - Stop all services"
    echo "  ${GREEN}restart${NC}     - Restart all services"
    echo "  ${GREEN}logs${NC}        - View logs (default: freqtrade)"
    echo "  ${GREEN}status${NC}      - Show status of all services"
    echo "  ${GREEN}research${NC}    - Open Jupyter Lab"
    echo "  ${GREEN}trade${NC}       - Start trading bot (dry-run)"
    echo "  ${RED}trade-live${NC}  - Start trading bot (LIVE - USE WITH CAUTION!)"
    echo "  ${GREEN}backtest${NC}    - Run backtest"
    echo "  ${GREEN}download${NC}    - Download historical data"
    echo "  ${GREEN}verify${NC}      - Verify data quality"
    echo "  ${YELLOW}clean${NC}       - Clean up containers and volumes"
    echo ""
    echo "Examples:"
    echo "  ./scripts/citadel.sh setup"
    echo "  ./scripts/citadel.sh download"
    echo "  ./scripts/citadel.sh research"
    echo "  ./scripts/citadel.sh backtest StoicEnsembleStrategy"
    echo ""
}

# Main command dispatcher
COMMAND=${1:-help}

case $COMMAND in
    setup)
        cmd_setup
        ;;
    start)
        cmd_start
        ;;
    stop)
        cmd_stop
        ;;
    restart)
        cmd_restart
        ;;
    logs)
        cmd_logs $2
        ;;
    status)
        cmd_status
        ;;
    research)
        cmd_research
        ;;
    trade)
        cmd_trade
        ;;
    trade-live)
        cmd_trade_live
        ;;
    backtest)
        cmd_backtest $2
        ;;
    download)
        cmd_download
        ;;
    verify)
        cmd_verify
        ;;
    clean)
        cmd_clean
        ;;
    help|--help|-h)
        cmd_help
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        echo ""
        cmd_help
        exit 1
        ;;
esac
