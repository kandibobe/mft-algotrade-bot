"""
Drift Analysis Engine
=====================

Compares live execution results with backtest expectations to detect model drift.
"""

import logging
from datetime import datetime, timedelta

from src.database.db_manager import DatabaseManager
from src.database.models import TradeRecord

logger = logging.getLogger(__name__)

class DriftAnalyzer:
    def __init__(self, threshold=0.10):
        self.threshold = threshold

    async def generate_daily_report(self):
        """Analyze performance drift over the last 24 hours."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=1)

        logger.info(f"Running Drift Analysis for period: {start_date} to {end_date}")

        # 1. Fetch live trades
        db = DatabaseManager()
        async with db.session() as session:
            live_trades = session.query(TradeRecord).filter(
                TradeRecord.entry_time >= start_date
            ).all()

        if not live_trades:
            return {"status": "no_data", "message": "No trades in the last 24h to analyze drift."}

        # 2. Compare with Backtest Expectation (Placeholder logic)
        # In a real system, we'd run a vectorized backtest on the same OHLCV data
        # and compare signal-by-signal or aggregate PnL
        live_pnl = sum([t.pnl_pct for t in live_trades if t.pnl_pct is not None])
        expected_pnl = 0.05 # Mocked expectation from latest WFO cycle

        drift = abs(live_pnl - expected_pnl)

        is_drifting = drift > self.threshold

        report = {
            "period": f"{start_date.date()} to {end_date.date()}",
            "trades_count": len(live_trades),
            "live_pnl_pct": round(live_pnl, 4),
            "expected_pnl_pct": round(expected_pnl, 4),
            "drift_magnitude": round(drift, 4),
            "is_drifting": is_drifting,
            "status": "CRITICAL" if is_drifting else "OK"
        }

        if is_drifting:
            logger.warning(f"🚨 MODEL DRIFT DETECTED: {drift:.2%} exceeds threshold {self.threshold:.2%}")

        return report
