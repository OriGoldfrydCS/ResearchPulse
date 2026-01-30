"""
Database connection and session management.

Uses SQLAlchemy with async support for PostgreSQL via DATABASE_URL.
Falls back gracefully when DATABASE_URL is not configured.
"""

from __future__ import annotations

import os
from typing import Optional, Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import QueuePool

# Base class for all models
Base = declarative_base()

# Global engine and session factory
_engine = None
_SessionLocal = None


def get_database_url() -> Optional[str]:
    """Get DATABASE_URL from environment."""
    return os.getenv("DATABASE_URL")


def is_database_configured() -> bool:
    """Check if DATABASE_URL is configured."""
    return bool(get_database_url())


def init_engine(database_url: Optional[str] = None) -> bool:
    """
    Initialize the database engine.
    
    Args:
        database_url: Optional override for DATABASE_URL
        
    Returns:
        True if engine was initialized, False otherwise
    """
    global _engine, _SessionLocal
    
    url = database_url or get_database_url()
    if not url:
        return False
    
    # Handle Supabase/Render postgres:// vs postgresql://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    
    try:
        _engine = create_engine(
            url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,  # Recycle connections after 30 min
            echo=os.getenv("SQL_ECHO", "false").lower() == "true",
        )
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
        return True
    except Exception as e:
        print(f"Failed to initialize database engine: {e}")
        return False


def get_engine():
    """Get the SQLAlchemy engine. Initializes if needed."""
    global _engine
    if _engine is None:
        init_engine()
    return _engine


def get_session_factory():
    """Get the session factory. Initializes engine if needed."""
    global _SessionLocal
    if _SessionLocal is None:
        init_engine()
    return _SessionLocal


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    
    Usage:
        with get_db_session() as db:
            db.query(Paper).all()
    """
    SessionLocal = get_session_factory()
    if SessionLocal is None:
        raise RuntimeError("Database not configured. Set DATABASE_URL environment variable.")
    
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI endpoints.
    
    Usage:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    SessionLocal = get_session_factory()
    if SessionLocal is None:
        raise RuntimeError("Database not configured. Set DATABASE_URL environment variable.")
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_connection() -> tuple[bool, str]:
    """
    Check database connection health.
    
    Returns:
        Tuple of (healthy: bool, message: str)
    """
    engine = get_engine()
    if engine is None:
        return False, "Database not configured"
    
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "Connected"
    except Exception as e:
        return False, str(e)


def create_all_tables():
    """Create all tables defined in models."""
    from . import models  # noqa: F401 - Import to register models
    
    engine = get_engine()
    if engine is None:
        raise RuntimeError("Database not configured. Set DATABASE_URL environment variable.")
    
    Base.metadata.create_all(bind=engine)


def drop_all_tables():
    """Drop all tables. USE WITH CAUTION."""
    engine = get_engine()
    if engine is None:
        raise RuntimeError("Database not configured. Set DATABASE_URL environment variable.")
    
    Base.metadata.drop_all(bind=engine)
