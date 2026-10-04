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
    return create_engine(get_database_url(), pool_pre_ping=True)
