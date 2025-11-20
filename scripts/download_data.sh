#!/bin/bash
# ==============================================================================
# Stoic Citadel - Data Download Script
# ==============================================================================
# Downloads historical OHLCV data for backtesting
#
# Usage:
#   ./scripts/download_data.sh [days] [timeframe]
#
# Examples:
#   ./scripts/download_data.sh 90 5m    # Download 90 days of 5-minute data
#   ./scripts/download_data.sh 365 1h   # Download 1 year of hourly data
# ==============================================================================

set -e  # Exit on error

# Configuration
DAYS=${1:-90}           # Default: 90 days
TIMEFRAME=${2:-5m}      # Default: 5 minutes
EXCHANGE="binance"

# Pairs to download
PAIRS=(
    "BTC/USDT"
    "ETH/USDT"
    "BNB/USDT"
    "SOL/USDT"
    "XRP/USDT"
    "ADA/USDT"
    "AVAX/USDT"
    "MATIC/USDT"
    "DOT/USDT"
    "LINK/USDT"
)

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║           STOIC CITADEL - DATA DOWNLOAD                            ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Exchange:    $EXCHANGE"
echo "📅 Days:        $DAYS"
echo "⏱️  Timeframe:   $TIMEFRAME"
echo "🔢 Pairs:       ${#PAIRS[@]}"
echo ""
echo "Starting download..."
echo ""

# Build pairs argument
PAIRS_ARG=""
for pair in "${PAIRS[@]}"; do
    PAIRS_ARG="$PAIRS_ARG --pairs $pair"
done

# Download data using Freqtrade
docker-compose run --rm freqtrade download-data \
    --exchange $EXCHANGE \
    $PAIRS_ARG \
    --timeframe $TIMEFRAME \
    --days $DAYS \
    --data-format-ohlcv json

echo ""
echo "✅ Download completed!"
echo ""
echo "Data location: ./user_data/data/$EXCHANGE/"
echo ""
echo "Next steps:"
echo "  1. Verify data: ./scripts/verify_data.sh"
echo "  2. Start research: docker-compose up jupyter"
echo "  3. Open notebook: http://localhost:8888 (token: stoic2024)"
echo ""
