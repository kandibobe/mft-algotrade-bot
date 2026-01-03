"""
Attribution Analysis & Persistence
==================================

Handles the storage and retrieval of the "Truth" (Signals + Executions).
Provides the bridge between runtime execution and post-trade analytics.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from src.database.db_manager import DatabaseManager
from src.database.models import TradeRecord, SignalRecord, ExecutionRecord

logger = logging.getLogger(__name__)

class AttributionService:
    """
    Service for persisting Signal and Execution records.
    Designed to be non-blocking and safe.
    """
    
    @staticmethod
    def record_signal(
        symbol: str,
        strategy: str,
        decision: Any, # StructuredTradeDecision
    ) -> Optional[int]:
        """
        Persist a signal. Returns signal_id.
        Should be called by the Strategy before order submission.
        """
        try:
            with DatabaseManager.session() as session:
                record = SignalRecord(
                    symbol=symbol,
                    strategy_name=strategy,
                    signal_type=decision.signal,
                    regime=decision.regime,
                    model_confidence=decision.confidence,
                    rsi=decision.metadata.get('rsi'),
                    ema_200=decision.metadata.get('ema_200'),
                    close_price=decision.metadata.get('close'),
                    meta_data=decision.metadata
                )
                session.add(record)
                session.flush() # Populate ID
                return record.id
        except Exception as e:
            logger.error(f"Failed to record signal: {e}")
            return None

    @staticmethod
    def record_execution(
        trade_data: Dict[str, Any],
        execution_metrics: Dict[str, Any],
        signal_id: Optional[int] = None,
        attribution_metadata: Optional[Dict] = None
    ):
        """
        Record a completed trade and its execution metrics.
        Should be called by SmartOrderExecutor upon fill.
        
        Args:
            trade_data: {symbol, exchange, side, price, amount, strategy}
            execution_metrics: {target_price, slippage_pct, latency_ms, spread_at_fill}
            signal_id: Optional ID of the signal that triggered this trade
            attribution_metadata: Context metadata carried by SmartOrder
        """
        try:
            with DatabaseManager.session() as session:
                # 1. Create TradeRecord (Accounting)
                trade = TradeRecord(
                    symbol=trade_data['symbol'],
                    exchange=trade_data.get('exchange', 'unknown'),
                    side=trade_data['side'],
                    entry_price=trade_data['price'],
                    amount=trade_data['amount'],
                    entry_time=datetime.utcnow(),
                    strategy_name=trade_data.get('strategy', 'unknown'),
                    meta_data=attribution_metadata
                )
                session.add(trade)
                session.flush() # get ID
                
                # 2. Link Signal if exists
                if signal_id:
                    signal = session.query(SignalRecord).get(signal_id)
                    if signal:
                        signal.trade_id = trade.id
                        # If SignalRecord had a different strategy name, trust the one in signal?
                        # Usually they match.
                
                # 3. Create ExecutionRecord (Engineering)
                exec_rec = ExecutionRecord(
                    trade_id=trade.id,
                    symbol=trade.symbol,
                    side=trade.side,
                    target_price=execution_metrics.get('target_price'),
                    fill_price=trade.entry_price,
                    slippage_pct=execution_metrics.get('slippage_pct'),
                    latency_ms=execution_metrics.get('latency_ms'),
                    spread_at_fill=execution_metrics.get('spread_at_fill'),
                    meta_data=execution_metrics.get('meta_data')
                )
                session.add(exec_rec)
                
                logger.info(f"Recorded Attribution for Trade {trade.id}: Slippage={exec_rec.slippage_pct:.4f}%")
        except Exception as e:
            logger.error(f"Failed to record execution attribution: {e}")
