"""Create the local AgriMind database and apply the initial schema."""

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv


DATABASE_NAME = "agrimind"
SCHEMA_PATH = Path(__file__).with_name("001_farmers_and_fields.sql")


def postgres_url(database_url: str, database_name: str) -> str:
    """Convert SQLAlchemy's Psycopg URL to a Psycopg connection URL."""
    normalized = database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    return f"{normalized.rsplit('/', 1)[0]}/{database_name}"


def main() -> None:
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured. Copy .env.example to .env.")

    admin_url = postgres_url(database_url, "postgres")
    with psycopg.connect(admin_url, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DATABASE_NAME,))
            if cursor.fetchone() is None:
                cursor.execute(f"CREATE DATABASE {DATABASE_NAME}")
                print(f"Created database: {DATABASE_NAME}")
            else:
                print(f"Database already exists: {DATABASE_NAME}")

    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    # This small bootstrap schema uses `--` comments. Remove them before
    # splitting statements so a semicolon in a comment is never executed.
    executable_schema = "\n".join(
        line for line in schema.splitlines() if not line.lstrip().startswith("--")
    )
    with psycopg.connect(postgres_url(database_url, DATABASE_NAME)) as connection:
        with connection.cursor() as cursor:
            for statement in executable_schema.split(";"):
                if statement.strip():
                    cursor.execute(statement)
        connection.commit()

    with psycopg.connect(postgres_url(database_url, DATABASE_NAME)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT farmer.name, array_agg(field.name ORDER BY field.name)
                FROM farmers AS farmer
                LEFT JOIN fields AS field ON field.farmer_id = farmer.id
                GROUP BY farmer.name
                ORDER BY farmer.name
                """
            )
            for farmer, fields in cursor.fetchall():
                print(f"{farmer}: {', '.join(fields)}")


if __name__ == "__main__":
    main()
