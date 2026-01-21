#!/bin/bash
# Stoic Citadel V7 - Quick Start Script

export PYTHONIOENCODING=utf-8

echo "🚀 Starting Stoic Citadel V7 System..."

# 1. Check for .env file
if [ ! -f .env ]; then
    echo "⚠️ .env file not found! Copying from .env.example..."
    cp .env.example .env
    echo "🚨 ACTION REQUIRED: Please edit .env and add your API Keys before running again."
    exit 1
fi

# 2. Build and Start Services
docker-compose up -d --build

echo "✅ System is starting in the background."
echo "📊 Dashboard: http://localhost:3000"
echo "⚙️  API: http://localhost:8080"
echo "📝 Logs: docker-compose logs -f freqtrade"