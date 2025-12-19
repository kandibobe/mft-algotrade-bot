"""
PostgreSQL Database Configuration with Connection Pooling
==========================================================

Optimized database configuration to prevent connection bottlenecks.

Usage:
    from config.database import get_engine, get_session

    # Get engine
    engine = get_engine()

    # Get session
    session = get_session()

    # Use session
    with session() as s:
        result = s.execute("SELECT * FROM trades")

Copyright (c) 2024-2025 Stoic Citadel
PROPRIETARY - All Rights Reserved
"""

import os
import logging
from typing import Optional
from sqlalchemy import create_engine, event, pool
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """
    PostgreSQL configuration with connection pooling.

    Implements:
    - QueuePool for connection reuse
    - Connection pre-ping to detect stale connections
    - Automatic connection recycling
    - Configurable pool size based on workload
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_recycle: int = 3600,
        pool_pre_ping: bool = True,
        pool_timeout: int = 30,
        echo: bool = False,
    ):
        """
        Initialize database configuration.

        Args:
            database_url: PostgreSQL connection string
                         (default: from env POSTGRES_URL or constructed from env vars)
            pool_size: Number of permanent connections (default: 10)
            max_overflow: Max temporary connections above pool_size (default: 20)
            pool_recycle: Recycle connections after N seconds (default: 3600 = 1h)
            pool_pre_ping: Test connections before use (default: True)
            pool_timeout: Max wait time for connection (default: 30s)
            echo: Log all SQL statements (default: False)
        """
        self.database_url = database_url or self._get_database_url()
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_recycle = pool_recycle
        self.pool_pre_ping = pool_pre_ping
        self.pool_timeout = pool_timeout
        self.echo = echo

        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None

    def _get_database_url(self) -> str:
        """
        Construct database URL from environment variables.

        Reads:
            POSTGRES_URL (complete URL) or
            POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST,
            POSTGRES_PORT, POSTGRES_DB
        """
        # Try complete URL first
        url = os.getenv('POSTGRES_URL')
        if url:
            return url

        # Construct from parts
        user = os.getenv('POSTGRES_USER', 'stoic_trader')
        password = os.getenv('POSTGRES_PASSWORD', '')
        host = os.getenv('POSTGRES_HOST', 'localhost')
        port = os.getenv('POSTGRES_PORT', '5433')
        database = os.getenv('POSTGRES_DB', 'trading_analytics')

        if not password:
            logger.warning("POSTGRES_PASSWORD not set - using empty password")

        url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        return url

    def get_engine(self) -> Engine:
        """
        Get or create SQLAlchemy engine with connection pooling.

        Returns:
            Configured SQLAlchemy engine
        """
        if self._engine is not None:
            return self._engine

        logger.info("Creating database engine with connection pooling")
        logger.info(f"Pool config: size={self.pool_size}, max_overflow={self.max_overflow}")

        self._engine = create_engine(
            self.database_url,
            # Connection pooling
            poolclass=pool.QueuePool,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_timeout=self.pool_timeout,
            pool_recycle=self.pool_recycle,
            pool_pre_ping=self.pool_pre_ping,
            # Performance
            echo=self.echo,
            echo_pool=False,  # Set to True for pool debugging
            # Isolation level for PostgreSQL
            isolation_level="READ COMMITTED",
            # Connection arguments
            connect_args={
                "connect_timeout": 10,
                "application_name": "stoic_citadel_bot",
                "options": "-c statement_timeout=30000"  # 30s query timeout
            }
        )

        # Add event listeners
        self._add_event_listeners(self._engine)

        logger.info("Database engine created successfully")
        return self._engine

    def _add_event_listeners(self, engine: Engine) -> None:
        """Add event listeners for monitoring and debugging."""

        @event.listens_for(engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            """Log new connections."""
            logger.debug("New database connection established")

        @event.listens_for(engine, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            """Log connection returns to pool."""
            logger.debug("Connection returned to pool")

        @event.listens_for(engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            """Log connection checkouts from pool."""
            logger.debug("Connection checked out from pool")

    def get_session_factory(self) -> sessionmaker:
        """
        Get or create session factory.

        Returns:
            SQLAlchemy sessionmaker
        """
        if self._session_factory is not None:
            return self._session_factory

        engine = self.get_engine()
        self._session_factory = sessionmaker(
            bind=engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False
        )

        return self._session_factory

    @contextmanager
    def get_session(self):
        """
        Get database session with automatic cleanup.

        Yields:
            SQLAlchemy session

        Example:
            with db_config.get_session() as session:
                result = session.execute("SELECT * FROM trades")
                session.commit()
        """
        factory = self.get_session_factory()
        session = factory()

        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()

    def get_pool_status(self) -> dict:
        """
        Get current connection pool status.

        Returns:
            Dict with pool statistics
        """
        if self._engine is None:
            return {"status": "not_initialized"}

        pool_obj = self._engine.pool

        return {
            "size": pool_obj.size(),
            "checked_in": pool_obj.checkedin(),
            "checked_out": pool_obj.checkedout(),
            "overflow": pool_obj.overflow(),
            "max_overflow": self.max_overflow,
            "pool_size": self.pool_size,
            "utilization_pct": (pool_obj.checkedout() / self.pool_size) * 100
        }

    def dispose(self) -> None:
        """Close all connections and dispose of the engine."""
        if self._engine is not None:
            logger.info("Disposing database engine and closing all connections")
            self._engine.dispose()
            self._engine = None
            self._session_factory = None


# Global instance
_db_config: Optional[DatabaseConfig] = None


def get_db_config(
    pool_size: int = 10,
    max_overflow: int = 20,
    reset: bool = False
) -> DatabaseConfig:
    """
    Get global database configuration instance.

    Args:
        pool_size: Connection pool size
        max_overflow: Max overflow connections
        reset: Force recreate the instance

    Returns:
        DatabaseConfig instance
    """
    global _db_config

    if _db_config is None or reset:
        _db_config = DatabaseConfig(
            pool_size=pool_size,
            max_overflow=max_overflow
        )

    return _db_config


# Convenience functions
def get_engine() -> Engine:
    """Get database engine (convenience function)."""
    return get_db_config().get_engine()


def get_session():
    """Get database session context manager (convenience function)."""
    return get_db_config().get_session()


def get_pool_status() -> dict:
    """Get connection pool status (convenience function)."""
    return get_db_config().get_pool_status()


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Initialize database with custom pool settings
    db_config = DatabaseConfig(
        pool_size=10,
        max_overflow=20,
        pool_recycle=3600,
        echo=True  # Log SQL for debugging
    )

    # Get engine
    engine = db_config.get_engine()
    print(f"Engine created: {engine}")

    # Check pool status
    status = db_config.get_pool_status()
    print(f"Pool status: {status}")

    # Use session
    with db_config.get_session() as session:
        result = session.execute("SELECT 1")
        print(f"Test query result: {result.fetchone()}")

    # Cleanup
    db_config.dispose()
