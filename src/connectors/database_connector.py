#!/usr/bin/env python3
"""
Database Connector
===================

Async PostgreSQL connector for trade analytics and logging.

Features:
- Trade history storage
- Performance analytics queries
- Position tracking
- Audit logging

Author: Stoic Citadel Team
License: MIT
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "trading_analytics"
    user: str = "stoic_trader"
    password: str = ""
    min_connections: int = 2
    max_connections: int = 10
    
    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class DatabaseConnector:
    """
    Async PostgreSQL database connector.
    
    Usage:
        config = DatabaseConfig(
            host="localhost",
            database="trading_analytics",
            user="stoic_trader",
            password="secret"
        )
        
        db = DatabaseConnector(config)
        await db.connect()
        
        # Log trade
        await db.log_trade(trade_data)
        
        # Query performance
        stats = await db.get_performance_stats(days=30)
    """
    
    # SQL Schema
    SCHEMA = """
    -- Trades table
    CREATE TABLE IF NOT EXISTS trades (
        id SERIAL PRIMARY KEY,
        trade_id VARCHAR(50) UNIQUE NOT NULL,
        symbol VARCHAR(20) NOT NULL,
        side VARCHAR(10) NOT NULL,
        quantity DECIMAL(20, 8) NOT NULL,
        entry_price DECIMAL(20, 8) NOT NULL,
        exit_price DECIMAL(20, 8),
        entry_time TIMESTAMP NOT NULL,
        exit_time TIMESTAMP,
        pnl DECIMAL(20, 8),
        pnl_pct DECIMAL(10, 4),
        fees DECIMAL(20, 8) DEFAULT 0,
        strategy VARCHAR(100),
        metadata JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Equity snapshots table
    CREATE TABLE IF NOT EXISTS equity_snapshots (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP NOT NULL,
        total_equity DECIMAL(20, 8) NOT NULL,
        cash_balance DECIMAL(20, 8) NOT NULL,
        positions_value DECIMAL(20, 8) NOT NULL,
        unrealized_pnl DECIMAL(20, 8) NOT NULL,
        realized_pnl DECIMAL(20, 8) NOT NULL,
        open_positions INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Signals log table
    CREATE TABLE IF NOT EXISTS signals_log (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP NOT NULL,
        symbol VARCHAR(20) NOT NULL,
        signal_type VARCHAR(20) NOT NULL,
        signal_value DECIMAL(10, 4),
        indicators JSONB,
        executed BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Alerts log table
    CREATE TABLE IF NOT EXISTS alerts_log (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP NOT NULL,
        alert_type VARCHAR(50) NOT NULL,
        severity VARCHAR(20) NOT NULL,
        message TEXT NOT NULL,
        metadata JSONB,
        acknowledged BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Create indexes
    CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
    CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
    CREATE INDEX IF NOT EXISTS idx_equity_timestamp ON equity_snapshots(timestamp);
    CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals_log(timestamp);
    CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts_log(timestamp);
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._pool = None
    
    async def connect(self):
        """Connect to database and create pool."""
        try:
            import asyncpg
            self._pool = await asyncpg.create_pool(
                self.config.dsn,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections
            )
            await self._init_schema()
            logger.info(f"Connected to PostgreSQL at {self.config.host}:{self.config.port}")
        except ImportError:
            logger.warning("asyncpg not installed. Using mock database.")
            self._pool = None
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            self._pool = None
    
    async def disconnect(self):
        """Close database connection."""
        if self._pool:
            await self._pool.close()
            logger.info("Disconnected from PostgreSQL")
    
    async def _init_schema(self):
        """Initialize database schema."""
        if self._pool:
            async with self._pool.acquire() as conn:
                await conn.execute(self.SCHEMA)
    
    @asynccontextmanager
    async def _get_connection(self) -> AsyncGenerator:
        """Get connection from pool."""
        if not self._pool:
            yield None
            return
        
        async with self._pool.acquire() as conn:
            yield conn
    
    # =========================================================================
    # Trade Operations
    # =========================================================================
    
    async def log_trade(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        entry_time: datetime,
        exit_price: Optional[float] = None,
        exit_time: Optional[datetime] = None,
        pnl: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        fees: float = 0,
        strategy: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Log a trade to database."""
        async with self._get_connection() as conn:
            if not conn:
                return
            
            await conn.execute("""
                INSERT INTO trades (
                    trade_id, symbol, side, quantity, entry_price, entry_time,
                    exit_price, exit_time, pnl, pnl_pct, fees, strategy, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ON CONFLICT (trade_id) DO UPDATE SET
                    exit_price = $7, exit_time = $8, pnl = $9, pnl_pct = $10
            """, trade_id, symbol, side, quantity, entry_price, entry_time,
                exit_price, exit_time, pnl, pnl_pct, fees, strategy,
                json.dumps(metadata) if metadata else None
            )
    
    async def update_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_time: datetime,
        pnl: float,
        pnl_pct: float
    ):
        """Update trade with exit information."""
        async with self._get_connection() as conn:
            if not conn:
                return
            
            await conn.execute("""
                UPDATE trades SET
                    exit_price = $2,
                    exit_time = $3,
                    pnl = $4,
                    pnl_pct = $5
                WHERE trade_id = $1
            """, trade_id, exit_price, exit_time, pnl, pnl_pct)
    
    async def get_trades(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get trade history."""
        async with self._get_connection() as conn:
            if not conn:
                return []
            
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            param_idx = 1
            
            if symbol:
                query += f" AND symbol = ${param_idx}"
                params.append(symbol)
                param_idx += 1
            
            if start_date:
                query += f" AND entry_time >= ${param_idx}"
                params.append(start_date)
                param_idx += 1
            
            if end_date:
                query += f" AND entry_time <= ${param_idx}"
                params.append(end_date)
                param_idx += 1
            
            query += f" ORDER BY entry_time DESC LIMIT ${param_idx}"
            params.append(limit)
            
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]
    
    # =========================================================================
    # Equity Snapshots
    # =========================================================================
    
    async def log_equity_snapshot(
        self,
        timestamp: datetime,
        total_equity: float,
        cash_balance: float,
        positions_value: float,
        unrealized_pnl: float,
        realized_pnl: float,
        open_positions: int
    ):
        """Log equity snapshot."""
        async with self._get_connection() as conn:
            if not conn:
                return
            
            await conn.execute("""
                INSERT INTO equity_snapshots (
                    timestamp, total_equity, cash_balance, positions_value,
                    unrealized_pnl, realized_pnl, open_positions
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, timestamp, total_equity, cash_balance, positions_value,
                unrealized_pnl, realized_pnl, open_positions)
    
    async def get_equity_curve(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get equity curve data."""
        async with self._get_connection() as conn:
            if not conn:
                return []
            
            query = "SELECT * FROM equity_snapshots WHERE 1=1"
            params = []
            param_idx = 1
            
            if start_date:
                query += f" AND timestamp >= ${param_idx}"
                params.append(start_date)
                param_idx += 1
            
            if end_date:
                query += f" AND timestamp <= ${param_idx}"
                params.append(end_date)
                param_idx += 1
            
            query += f" ORDER BY timestamp ASC LIMIT ${param_idx}"
            params.append(limit)
            
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]
    
    # =========================================================================
    # Analytics Queries
    # =========================================================================
    
    async def get_performance_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get performance statistics."""
        async with self._get_connection() as conn:
            if not conn:
                return {}
            
            start_date = datetime.now() - timedelta(days=days)
            
            # Trade statistics
            trade_stats = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_trades,
                    COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_trades,
                    COUNT(CASE WHEN pnl < 0 THEN 1 END) as losing_trades,
                    COALESCE(SUM(pnl), 0) as total_pnl,
                    COALESCE(AVG(pnl), 0) as avg_pnl,
                    COALESCE(MAX(pnl), 0) as best_trade,
                    COALESCE(MIN(pnl), 0) as worst_trade,
                    COALESCE(SUM(fees), 0) as total_fees
                FROM trades
                WHERE exit_time IS NOT NULL
                AND entry_time >= $1
            """, start_date)
            
            # Calculate derived metrics
            total = trade_stats['total_trades'] or 0
            wins = trade_stats['winning_trades'] or 0
            
            return {
                "period_days": days,
                "total_trades": total,
                "winning_trades": wins,
                "losing_trades": trade_stats['losing_trades'] or 0,
                "win_rate": (wins / total * 100) if total > 0 else 0,
                "total_pnl": float(trade_stats['total_pnl'] or 0),
                "avg_pnl_per_trade": float(trade_stats['avg_pnl'] or 0),
                "best_trade": float(trade_stats['best_trade'] or 0),
                "worst_trade": float(trade_stats['worst_trade'] or 0),
                "total_fees": float(trade_stats['total_fees'] or 0)
            }
    
    async def get_performance_by_symbol(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get performance breakdown by symbol."""
        async with self._get_connection() as conn:
            if not conn:
                return []
            
            start_date = datetime.now() - timedelta(days=days)
            
            rows = await conn.fetch("""
                SELECT
                    symbol,
                    COUNT(*) as trades,
                    COUNT(CASE WHEN pnl > 0 THEN 1 END) as wins,
                    COALESCE(SUM(pnl), 0) as total_pnl,
                    COALESCE(AVG(pnl), 0) as avg_pnl
                FROM trades
                WHERE exit_time IS NOT NULL
                AND entry_time >= $1
                GROUP BY symbol
                ORDER BY total_pnl DESC
            """, start_date)
            
            return [dict(row) for row in rows]
    
    async def get_daily_pnl(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get daily P&L breakdown."""
        async with self._get_connection() as conn:
            if not conn:
                return []
            
            start_date = datetime.now() - timedelta(days=days)
            
            rows = await conn.fetch("""
                SELECT
                    DATE(exit_time) as date,
                    COUNT(*) as trades,
                    COALESCE(SUM(pnl), 0) as pnl,
                    COUNT(CASE WHEN pnl > 0 THEN 1 END) as wins
                FROM trades
                WHERE exit_time IS NOT NULL
                AND entry_time >= $1
                GROUP BY DATE(exit_time)
                ORDER BY date DESC
            """, start_date)
            
            return [dict(row) for row in rows]
    
    # =========================================================================
    # Signals & Alerts Logging
    # =========================================================================
    
    async def log_signal(
        self,
        symbol: str,
        signal_type: str,
        signal_value: float,
        indicators: Optional[Dict] = None,
        executed: bool = False
    ):
        """Log trading signal."""
        async with self._get_connection() as conn:
            if not conn:
                return
            
            await conn.execute("""
                INSERT INTO signals_log (
                    timestamp, symbol, signal_type, signal_value, indicators, executed
                ) VALUES ($1, $2, $3, $4, $5, $6)
            """, datetime.now(), symbol, signal_type, signal_value,
                json.dumps(indicators) if indicators else None, executed)
    
    async def log_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        metadata: Optional[Dict] = None
    ):
        """Log system alert."""
        async with self._get_connection() as conn:
            if not conn:
                return
            
            await conn.execute("""
                INSERT INTO alerts_log (
                    timestamp, alert_type, severity, message, metadata
                ) VALUES ($1, $2, $3, $4, $5)
            """, datetime.now(), alert_type, severity, message,
                json.dumps(metadata) if metadata else None)
    
    # =========================================================================
    # Health Check
    # =========================================================================
    
    async def health_check(self) -> Dict[str, Any]:
        """Check database health."""
        async with self._get_connection() as conn:
            if not conn:
                return {
                    "service": "database",
                    "status": "disconnected",
                    "connected": False
                }
            
            try:
                result = await conn.fetchval("SELECT 1")
                
                # Get table sizes
                sizes = await conn.fetch("""
                    SELECT relname as table, n_live_tup as rows
                    FROM pg_stat_user_tables
                    ORDER BY n_live_tup DESC
                """)
                
                return {
                    "service": "database",
                    "status": "healthy",
                    "connected": True,
                    "tables": {row['table']: row['rows'] for row in sizes}
                }
            except Exception as e:
                return {
                    "service": "database",
                    "status": "error",
                    "error": str(e)
                }
