"""Central PostgreSQL connection configuration for AgriMind."""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def get_database_url() -> str:
    """Return a Psycopg-compatible PostgreSQL connection URL."""

    load_dotenv()

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not configured."
        )

    return database_url.replace(
        "postgresql+psycopg://",
        "postgresql://",
        1,
    )


def create_database_engine() -> Engine:
    """Create a reusable SQLAlchemy engine with inexpensive connection health checks."""
    """
    PostgreSQL
    ↓
    connection was idle for a long time
    ↓
    database/network closes it
    ↓
    SQLAlchemy still has that connection in pool
    Without pool_pre_ping=True, your application might try to use that dead connection and get an error.
    """
    return create_engine(get_database_url(), pool_pre_ping=True)


    """
    Need DB connection
       ↓
    Get connection from pool
       ↓
    Is connection alive?
      ↙       ↘
    YES        NO
     ↓          ↓
   Use it    Reconnect
    """
