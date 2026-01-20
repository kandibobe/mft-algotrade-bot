"""
Order Ledger - Persistent Order Storage
========================================

Provides persistent storage for all orders with idempotency guarantees.

Key Features:
1. SQLite-based order history (production-ready)
2. Idempotency key support (prevents duplicate orders)
3. Order state recovery after restart
4. Atomic operations with transaction support

Critical for Production:
- Prevents duplicate orders on restart/retry
- Full audit trail of all trading activity
- Enables post-mortem analysis
- Required for regulatory compliance

Usage:
    ledger = OrderLedger("orders.db")

    # Check idempotency before sending order
    if ledger.is_duplicate(idempotency_key="my_unique_key"):
        logger.warning("Order already submitted")
        return

    # Store order
    ledger.store_order(order, idempotency_key="my_unique_key")

    # Update order status
    ledger.update_order_status(order_id, OrderStatus.FILLED)

    # Recover state after restart
    active_orders = ledger.get_active_orders()
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

try:
    from src.order_manager.models import Order, OrderSide, OrderStatus, OrderType
except ImportError:
    # Fallback for testing
    pass
    Order = Any  # type: ignore
    OrderStatus = Any  # type: ignore

logger = logging.getLogger(__name__)


class OrderLedger:
    """
    Persistent order storage with idempotency guarantees.

    Thread-safe SQLite database for storing all orders.

    Schema:
        orders table:
            - order_id: PRIMARY KEY
            - idempotency_key: UNIQUE (for duplicate detection)
            - client_order_id: TEXT
            - exchange_order_id: TEXT
            - symbol: TEXT
            - order_type: TEXT
            - side: TEXT (buy/sell)
            - quantity: REAL
            - price: REAL
            - status: TEXT
            - created_at: TIMESTAMP
            - updated_at: TIMESTAMP
            - order_data: JSON (full order object)
    """

    def __init__(self, db_path: str = "data/orders.db"):
        """
        Initialize order ledger.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._create_tables()
        logger.info(f"Order ledger initialized: {self.db_path}")

    @contextmanager
    def _get_connection(self):
        """Get database connection with automatic commit/rollback."""
        conn = sqlite3.connect(
            self.db_path,
            isolation_level="DEFERRED",  # Transaction support
            check_same_thread=False,  # Allow multi-threading
        )
        conn.row_factory = sqlite3.Row  # Return dict-like rows
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _create_tables(self):
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    idempotency_key TEXT UNIQUE,
                    client_order_id TEXT,
                    exchange_order_id TEXT,
                    symbol TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    order_data TEXT NOT NULL
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS order_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT NOT NULL,
                    old_status TEXT,
                    new_status TEXT NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    update_data TEXT,
                    FOREIGN KEY (order_id) REFERENCES orders(order_id)
                )
            """
            )

            # Create indexes separately for compatibility
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON orders (status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON orders (symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON orders (created_at)")

        logger.info("Database tables created/verified")

    def is_duplicate(self, idempotency_key: str) -> bool:
        """
        Check if order with this idempotency key already exists.

        Args:
            idempotency_key: Unique key for this order

        Returns:
            True if order already exists, False otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM orders WHERE idempotency_key = ? LIMIT 1", (idempotency_key,)
            )
            return cursor.fetchone() is not None

    def store_order(
        self,
        order: Any,  # Order object
        idempotency_key: str | None = None,
    ) -> bool:
        """
        Store order in ledger.

        Args:
            order: Order object to store
            idempotency_key: Optional idempotency key (defaults to order_id)

        Returns:
            True if stored successfully, False if duplicate
        """
        idempotency_key = idempotency_key or order.order_id

        # Convert order to dict for JSON storage
        if hasattr(order, "__dict__"):
            order_dict = {
                k: str(v) if isinstance(v, (datetime, type(None))) else v
                for k, v in order.__dict__.items()
            }
        else:
            order_dict = dict(order)

        order_json = json.dumps(order_dict, default=str)

        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO orders (
                        order_id, idempotency_key, client_order_id,
                        exchange_order_id, symbol, order_type, side,
                        quantity, price, status, created_at, updated_at,
                        order_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        order.order_id,
                        idempotency_key,
                        getattr(order, "client_order_id", None),
                        getattr(order, "exchange_order_id", None),
                        order.symbol,
                        str(order.order_type),
                        str(order.side),
                        order.quantity,
                        getattr(order, "price", None),
                        str(order.status),
                        getattr(order, "created_at", datetime.now()).isoformat(),
                        datetime.now().isoformat(),
                        order_json,
                    ),
                )

            logger.info(f"Stored order {order.order_id} with idempotency key {idempotency_key}")
            return True

        except sqlite3.IntegrityError as e:
            if "idempotency_key" in str(e):
                logger.warning(f"Duplicate order detected: {idempotency_key}")
                return False
            raise

    def update_order_status(
        self,
        order_id: str,
        new_status: str,
        update_data: dict | None = None,
    ):
        """
        Update order status and record change.

        Args:
            order_id: Order ID to update
            new_status: New order status
            update_data: Optional additional data
        """
        with self._get_connection() as conn:
            # Get current status
            cursor = conn.execute("SELECT status FROM orders WHERE order_id = ?", (order_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Order {order_id} not found")

            old_status = row["status"]

            # Update order
            conn.execute(
                """
                UPDATE orders
                SET status = ?, updated_at = ?
                WHERE order_id = ?
            """,
                (new_status, datetime.now().isoformat(), order_id),
            )

            # Record update history
            conn.execute(
                """
                INSERT INTO order_updates (
                    order_id, old_status, new_status, updated_at, update_data
                ) VALUES (?, ?, ?, ?, ?)
            """,
                (
                    order_id,
                    old_status,
                    new_status,
                    datetime.now().isoformat(),
                    json.dumps(update_data or {}),
                ),
            )

        logger.info(f"Updated order {order_id}: {old_status} → {new_status}")

    def get_order(self, order_id: str) -> dict | None:
        """
        Retrieve order by ID.

        Args:
            order_id: Order ID to retrieve

        Returns:
            Order dict or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
            row = cursor.fetchone()

            if row:
                order_dict = dict(row)
                order_dict["order_data"] = json.loads(order_dict["order_data"])
                return order_dict
            return None

    def get_active_orders(self) -> list[dict]:
        """
        Get all active orders (pending, partially filled).

        Returns:
            List of active order dicts
        """
        active_statuses = ["pending", "open", "partially_filled"]

        with self._get_connection() as conn:
            cursor = conn.execute(
                f"SELECT * FROM orders WHERE status IN ({','.join('?' * len(active_statuses))})",
                active_statuses,
            )

            orders = []
            for row in cursor.fetchall():
                order_dict = dict(row)
                order_dict["order_data"] = json.loads(order_dict["order_data"])
                orders.append(order_dict)

            return orders

    def get_orders_by_symbol(self, symbol: str, limit: int = 100) -> list[dict]:
        """
        Get recent orders for a symbol.

        Args:
            symbol: Trading symbol
            limit: Max number of orders to return

        Returns:
            List of order dicts
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM orders
                WHERE symbol = ?
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (symbol, limit),
            )

            orders = []
            for row in cursor.fetchall():
                order_dict = dict(row)
                order_dict["order_data"] = json.loads(order_dict["order_data"])
                orders.append(order_dict)

            return orders

    def get_order_history(self, order_id: str) -> list[dict]:
        """
        Get update history for an order.

        Args:
            order_id: Order ID

        Returns:
            List of update records
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM order_updates
                WHERE order_id = ?
                ORDER BY updated_at ASC
            """,
                (order_id,),
            )

            history = []
            for row in cursor.fetchall():
                update_dict = dict(row)
                update_dict["update_data"] = json.loads(update_dict["update_data"])
                history.append(update_dict)

            return history

    def get_statistics(self, start_date: datetime | None = None) -> dict:
        """
        Get order statistics.

        Args:
            start_date: Optional start date for filtering

        Returns:
            Statistics dict
        """
        where_clause = ""
        params = []

        if start_date:
            where_clause = "WHERE created_at >= ?"
            params.append(start_date.isoformat())

        with self._get_connection() as conn:
            # Total orders
            cursor = conn.execute(f"SELECT COUNT(*) as count FROM orders {where_clause}", params)
            total_orders = cursor.fetchone()["count"]

            # Orders by status
            cursor = conn.execute(
                f"""
                SELECT status, COUNT(*) as count
                FROM orders {where_clause}
                GROUP BY status
            """,
                params,
            )
            by_status = {row["status"]: row["count"] for row in cursor.fetchall()}

            # Orders by symbol
            cursor = conn.execute(
                f"""
                SELECT symbol, COUNT(*) as count
                FROM orders {where_clause}
                GROUP BY symbol
                ORDER BY count DESC
                LIMIT 10
            """,
                params,
            )
            by_symbol = {row["symbol"]: row["count"] for row in cursor.fetchall()}

            return {
                "total_orders": total_orders,
                "by_status": by_status,
                "by_symbol": by_symbol,
            }

    def cleanup_old_orders(self, days: int = 90):
        """
        Archive or delete old completed orders.

        Args:
            days: Keep orders from last N days
        """
        cutoff_date = datetime.now() - timedelta(days=days)  # type: ignore

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM orders
                WHERE status IN ('filled', 'canceled', 'rejected', 'expired')
                AND updated_at < ?
            """,
                (cutoff_date.isoformat(),),
            )

            deleted_count = cursor.rowcount

        logger.info(f"Cleaned up {deleted_count} old orders (older than {days} days)")
        return deleted_count


# Convenience function
def create_idempotency_key(
    symbol: str, side: str, quantity: float, timestamp: datetime | None = None
) -> str:
    """
    Generate idempotency key for an order.

    Args:
        symbol: Trading symbol
        side: buy/sell
        quantity: Order quantity
        timestamp: Optional timestamp (defaults to now)

    Returns:
        Unique idempotency key
    """
    import hashlib

    ts = timestamp or datetime.now()
    data = f"{symbol}_{side}_{quantity}_{ts.isoformat()}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]