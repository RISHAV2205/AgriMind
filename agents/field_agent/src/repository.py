"""PostgreSQL repository for field sensor readings."""

import psycopg

from database.connection import get_database_url
from agents.field_agent.src.interface import FieldReading


class FieldReadingRepository:
    """Store and retrieve field readings from PostgreSQL."""

    def __init__(self, database_url: str | None = None) -> None:
        self._database_url = database_url or get_database_url()

        if not self._database_url:
            raise ValueError(
                "DATABASE_URL environment variable is not configured."
            )

    def save(self, reading: FieldReading) -> int:
        """Insert one field reading and return its database ID."""

        query = """
            INSERT INTO field_readings (
                field_id,
                soil_moisture_percent,
                soil_temperature_c,
                soil_ph,
                nitrogen_mg_kg,
                phosphorus_mg_kg,
                potassium_mg_kg,
                recorded_at
            )
            VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s
            )
            RETURNING id;
        """

        with psycopg.connect(self._database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (
                        reading.field_id,
                        reading.soil_moisture_percent,
                        reading.soil_temperature_c,
                        reading.soil_ph,
                        reading.nitrogen_mg_kg,
                        reading.phosphorus_mg_kg,
                        reading.potassium_mg_kg,
                        reading.timestamp,
                    ),
                )

                reading_id = cursor.fetchone()[0]

            connection.commit()

        return reading_id

    def get_latest(self, field_id: str) -> FieldReading | None:
        """Return the latest stored reading for a field."""

        query = """
            SELECT
                field_id,
                soil_moisture_percent,
                soil_temperature_c,
                soil_ph,
                nitrogen_mg_kg,
                phosphorus_mg_kg,
                potassium_mg_kg,
                recorded_at
            FROM field_readings
            WHERE field_id = %s
            ORDER BY recorded_at DESC
            LIMIT 1;
        """

        with psycopg.connect(self._database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (field_id,))
                row = cursor.fetchone()

        if row is None:
            return None

        return FieldReading(
            field_id=str(row[0]),
            soil_moisture_percent=float(row[1]),
            soil_temperature_c=float(row[2]),
            soil_ph=float(row[3]),
            nitrogen_mg_kg=float(row[4]),
            phosphorus_mg_kg=float(row[5]),
            potassium_mg_kg=float(row[6]),
            timestamp=row[7],
        )