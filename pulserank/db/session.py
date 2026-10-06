"""Database session and engine management."""

import logging
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import OperationalError

from pulserank.config import settings
from pulserank.db.base import Base

logger = logging.getLogger(__name__)


def create_db_engine():
    """Create database engine with fallback to SQLite if PostgreSQL is unreachable."""
    database_url = settings.database_url
    try:
        if database_url.startswith("sqlite"):
            engine = create_engine(
                database_url,
                connect_args={"check_same_thread": False},
                echo=False,
            )
        else:
            engine = create_engine(
                database_url,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                echo=False,
            )
            # Test connectivity
            with engine.connect() as conn:
                pass
        return engine
    except (OperationalError, Exception) as exc:
        logger.warning(
            f"Primary database connection failed ({exc}). Falling back to SQLite at {settings.test_database_url}"
        )
        engine = create_engine(
            settings.test_database_url,
            connect_args={"check_same_thread": False},
            echo=False,
        )
        return engine


engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency for obtaining a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
