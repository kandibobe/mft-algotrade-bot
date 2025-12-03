-- Stoic Citadel - PostgreSQL Setup Script
-- Run this after first launch to create analytics tables

-- Performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    win_rate DECIMAL(5,2),
    total_profit DECIMAL(15,2),
    avg_profit DECIMAL(5,2),
    avg_winning_trade DECIMAL(5,2),
    avg_losing_trade DECIMAL(5,2),
    max_profit DECIMAL(5,2),
    max_loss DECIMAL(5,2),
    profit_factor DECIMAL(5,2),
    avg_duration DECIMAL(10,2),
    max_duration DECIMAL(10,2)
);

-- Strategy metadata
CREATE TABLE IF NOT EXISTS strategy_versions (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    deployed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    config JSONB,
    notes TEXT
);

-- Market regime tracking
CREATE TABLE IF NOT EXISTS market_regimes (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    btc_price DECIMAL(15,2),
    regime VARCHAR(20),  -- 'bull', 'bear', 'neutral'
    ema_200 DECIMAL(15,2),
    volatility DECIMAL(5,2)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_strategy_name ON strategy_versions(strategy_name);
CREATE INDEX IF NOT EXISTS idx_regime_timestamp ON market_regimes(timestamp);

-- Insert initial strategy version
INSERT INTO strategy_versions (strategy_name, version, notes)
VALUES ('StoicStrategyV1', '1.1.0', 'Enhanced version with improved regime detection')
ON CONFLICT DO NOTHING;

COMMIT;

-- Success message
SELECT 'PostgreSQL setup completed successfully!' AS status;