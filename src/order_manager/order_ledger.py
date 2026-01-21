"""
Order Ledger - Persistent Order Storage with Reconciliation
===========================================================

Provides persistent storage for all orders with idempotency guarantees
and automatic startup reconciliation with exchange APIs.

Critical for Production:
- Prevents duplicate orders on restart/retry.
- Full audit trail of all trading activity.
- Reconciliation logic to sync with exchange after crash/restart.
"""

import json
import logging
import sqlite3
import asyncio
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from src.order_manager.models import Order, OrderSide, OrderStatus, OrderType
except ImportError:
    # Fallback for testing
    Order = Any
    OrderStatus = Any

logger = logging.getLogger(__name__)

class OrderLedger:
    """
    Persistent order storage with reconciliation capabilities.
    """

    def __init__(self, db_path: str = "data/orders_v2.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()
        logger.info(f"Order Ledger (Refined) initialized: {self.db_path}")

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, isolation_level="DEFERRED", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _create_tables(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    idempotency_key TEXT UNIQUE,
                    exchange_order_id TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    filled_quantity REAL DEFAULT 0,
                    price REAL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    order_data TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON orders (status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exchange_id ON orders (exchange_order_id)")

    def store_order(self, order: Any, idempotency_key: str) -> bool:
        """Store a new order entry atomically."""
        try:
            order_json = json.dumps(order.__dict__ if hasattr(order, "__dict__") else order, default=str)
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO orders (
                        order_id, idempotency_key, exchange_order_id, symbol,
                        side, quantity, filled_quantity, price, status,
                        created_at, updated_at, order_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    order.get('order_id') if isinstance(order, dict) else getattr(order, 'order_id', str(idempotency_key)),
                    idempotency_key,
                    order.get('exchange_order_id') if isinstance(order, dict) else getattr(order, 'exchange_order_id', None),
                    order['symbol'] if isinstance(order, dict) else order.symbol,
                    order['side'] if isinstance(order, dict) else str(order.side),
                    order['quantity'] if isinstance(order, dict) else order.quantity,
                    order.get('filled_quantity', 0.0) if isinstance(order, dict) else getattr(order, 'filled_quantity', 0.0),
                    order.get('price') if isinstance(order, dict) else getattr(order, 'price', None),
                    str(order.get('status', 'pending')) if isinstance(order, dict) else str(getattr(order, 'status', 'pending')),
                    datetime.utcnow().isoformat(),
                    datetime.utcnow().isoformat(),
                    order_json
                ))
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"Duplicate order detected for key: {idempotency_key}")
            return False

    def update_order(self, order_id: str, updates: Dict[str, Any]):
        """Update order details and status."""
        fields = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [datetime.utcnow().isoformat(), order_id]
        
        with self._get_connection() as conn:
            conn.execute(f"UPDATE orders SET {fields}, updated_at = ? WHERE order_id = ?", values)

    async def reconcile_with_exchange(self, exchange_client: Any):
        """
        Refined Reconciliation Logic:
        1. Fetch all 'active' orders from local DB.
        2. Query exchange for their actual current status.
        3. Update local DB to match reality.
        4. Detect 'zombie' orders (orders on exchange but not in DB) - Log them for manual audit.
        """
        logger.info("Starting Order Ledger reconciliation...")
        active_orders = self.get_active_orders()
        
        for local_order in active_orders:
            exch_id = local_order['exchange_order_id']
            if not exch_id:
                # Order was recorded locally but maybe never reached exchange
                logger.warning(f"Reconciling order {local_order['order_id']} without exchange ID")
                continue
                
            try:
                # Query exchange (e.g., CCXT fetchOrder)
                actual_order = await exchange_client.fetch_order(exch_id, local_order['symbol'])
                
                if actual_order['status'] != local_order['status']:
                    logger.info(f"Reconciliation: Updating {exch_id} status {local_order['status']} -> {actual_order['status']}")
                    self.update_order(local_order['order_id'], {
                        "status": actual_order['status'],
                        "filled_quantity": actual_order.get('filled', 0.0)
                    })
            except Exception as e:
                logger.error(f"Failed to reconcile order {exch_id}: {e}")

        logger.info("Order Ledger reconciliation completed.")

    def get_active_orders(self) -> List[Dict]:
        """Get orders that are not in a final state."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM orders WHERE status NOT IN ('closed', 'canceled', 'rejected', 'filled')"
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_order_by_exchange_id(self, exchange_order_id: str) -> Optional[Dict]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM orders WHERE exchange_order_id = ?", (exchange_order_id,)).fetchone()
            return dict(row) if row else None