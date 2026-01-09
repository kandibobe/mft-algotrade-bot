"""
Database Manager
================

Centralizes database connections and session management.
Uses Unified Configuration for connection details.
"""

import logging
import os
import sys
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine, pool
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from src.config.manager import config

logger = logging.getLogger(__name__)

Base = declarative_base()


class DatabaseManager:
    _engine: Any | None = None
    _session_factory: sessionmaker | None = None

    @classmethod
    def get_engine(cls):
        """Lazy initialization of SQLAlchemy engine."""
        if cls._engine is None:
            cfg = config()
            # Default to SQLite for local development if PG not configured
            db_url = os.getenv("DATABASE_URL")

            if not db_url:
                # Construct from env vars or use SQLite
                user = os.getenv("POSTGRES_USER")
                if user:
                    password = os.getenv("POSTGRES_PASSWORD", "")
                    host = os.getenv("POSTGRES_HOST", "localhost")
                    port = os.getenv("POSTGRES_PORT", "5432")
                    db_name = os.getenv("POSTGRES_DB", "stoic_citadel")
                    db_url = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
                else:
                    db_url = cfg.paths.db_url
                    logger.info(f"Using SQLite database: {db_url}")

            logger.info(
                f"Initializing database engine: {db_url.split('@')[-1]}"
            )  # Log without credentials

            # BRUTE-FORCE FIX: Force 127.0.0.1 instead of localhost for Windows compatibility
            if "localhost" in db_url:
                db_url = db_url.replace("localhost", "127.0.0.1")

            logger.info(f"Connecting to DB at: {db_url}")

            kwargs = {
                "pool_pre_ping": True,
            }
            
            if db_url.startswith("postgresql"):
                kwargs["poolclass"] = pool.QueuePool
                kwargs["pool_size"] = 10
                kwargs["max_overflow"] = 20
            else:
                kwargs["poolclass"] = pool.StaticPool

            cls._engine = create_engine(db_url, **kwargs)

        return cls._engine

    @classmethod
    def get_url(cls) -> str:
        """Get database URL for Alembic."""
        # Force initialization to get URL construction logic
        cls.get_engine()
        # access protected attribute to get the final URL
        return str(cls._engine.url)

    @classmethod
    def get_session_factory(cls) -> sessionmaker:
        """Lazy initialization of session factory."""
        if cls._session_factory is None:
            engine = cls.get_engine()
            cls._session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
        return cls._session_factory

    @classmethod
    @contextmanager
    def session(cls) -> Session:
        """Context manager for database sessions."""
        factory = cls.get_session_factory()
        session = factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    @classmethod
    def init_db(cls):
        """Create all tables."""
        engine = cls.get_engine()
        Base.metadata.create_all(engine)
        logger.info("Database tables initialized.")
