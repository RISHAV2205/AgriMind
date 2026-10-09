"""PostgreSQL repository for farmers and their fields.

Kept separate from `FieldReadingRepository` so that read-only farm/field
queries used by the dashboard do not depend on the sensor-reading schema.
"""

import psycopg

from database.connection import get_database_url


class FarmRepository:
    """Read farmers and the fields they own."""

    def __init__(self, database_url: str | None = None) -> None:
        self._database_url = database_url or get_database_url()

        if not self._database_url:
            raise ValueError("DATABASE_URL environment variable is not configured.")

    def list_farmers(self) -> list[dict]:
        """Return every farmer with the number of fields they own."""

        query = """
            SELECT
                farmer.id,
                farmer.name,
                COUNT(field.id) AS field_count
            FROM farmers AS farmer
            LEFT JOIN fields AS field ON field.farmer_id = farmer.id
            GROUP BY farmer.id, farmer.name
            ORDER BY farmer.name;
        """

        with psycopg.connect(self._database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()

        return [
            {"id": row[0], "name": row[1], "field_count": row[2]}
            for row in rows
        ]

    def list_fields(self, farmer_id: int) -> list[dict]:
        """Return the fields owned by one farmer, ordered by name."""

        query = """
            SELECT id, name, farmer_id
            FROM fields
            WHERE farmer_id = %s
            ORDER BY name;
        """

        with psycopg.connect(self._database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (farmer_id,))
                rows = cursor.fetchall()

        return [
            {"id": row[0], "name": row[1], "farmer_id": row[2]}
            for row in rows
        ]

    def get_field(self, field_id: int) -> dict | None:
        """Return one field, or None when it does not exist."""

        query = """
            SELECT id, name, farmer_id
            FROM fields
            WHERE id = %s;
        """

        with psycopg.connect(self._database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (field_id,))
                row = cursor.fetchone()

        if row is None:
            return None

        return {"id": row[0], "name": row[1], "farmer_id": row[2]}
