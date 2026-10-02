"""In-memory, realistic soil sensor simulator for local development."""

from __future__ import annotations

from datetime import datetime, timezone
import random

from agents.field_agent.src.interface import FieldReading


class FieldDataSimulator:
    """Generate bounded random-walk soil readings, separately for each field.

    Values evolve from the prior reading rather than being independently random.
    This makes moisture and nutrient trends suitable for future risk detection.
    """

    _INITIAL_RANGES = {
        "soil_moisture_percent": (35.0, 65.0),
        "soil_temperature_c": (20.0, 31.0),
        "soil_ph": (5.8, 7.0),
        "nitrogen_mg_kg": (45.0, 85.0),
        "phosphorus_mg_kg": (20.0, 45.0),
        "potassium_mg_kg": (130.0, 230.0),
    }

    _BOUNDS = {
        "soil_moisture_percent": (5.0, 90.0),
        "soil_temperature_c": (10.0, 45.0),
        "soil_ph": (4.5, 8.5),
        "nitrogen_mg_kg": (5.0, 140.0),
        "phosphorus_mg_kg": (2.0, 90.0),
        "potassium_mg_kg": (30.0, 400.0),
    }

    def __init__(self, seed: int | None = None) -> None:
        self._random = random.Random(seed)
        self._latest_by_field: dict[str, FieldReading] = {}

    def get_reading(self, field_id: str) -> FieldReading:
        """Return the next soil snapshot for a field's evolving random walk."""
        cleaned_field_id = field_id.strip()
        if not cleaned_field_id:
            raise ValueError("field_id is required.")

        previous = self._latest_by_field.get(cleaned_field_id)
        reading = (
            self._initial_reading(cleaned_field_id)
            if previous is None
            else self._next_reading(previous)
        )
        self._latest_by_field[cleaned_field_id] = reading
        return reading

    def _initial_reading(self, field_id: str) -> FieldReading:
        values = {
            name: self._random.uniform(*value_range)
            for name, value_range in self._INITIAL_RANGES.items()
        }
        return self._to_reading(field_id, values)

    def _next_reading(self, previous: FieldReading) -> FieldReading:
        # Moisture generally falls between irrigations, with small natural noise.
        moisture_change = -self._random.uniform(0.10, 0.55) + self._random.gauss(0, 0.12)
        values = {
            "soil_moisture_percent": previous.soil_moisture_percent + moisture_change,
            "soil_temperature_c": previous.soil_temperature_c + self._random.gauss(0, 0.35),
            "soil_ph": previous.soil_ph + self._random.gauss(0, 0.012),
            "nitrogen_mg_kg": previous.nitrogen_mg_kg - self._random.uniform(0.01, 0.12),
            "phosphorus_mg_kg": previous.phosphorus_mg_kg - self._random.uniform(0.005, 0.06),
            "potassium_mg_kg": previous.potassium_mg_kg - self._random.uniform(0.01, 0.10),
        }
        return self._to_reading(previous.field_id, values)

    def _to_reading(self, field_id: str, values: dict[str, float]) -> FieldReading:
        bounded = {
            name: round(max(lower, min(value, upper)), 2)
            for name, value in values.items()
            for lower, upper in [self._BOUNDS[name]]
        }
        return FieldReading(
            field_id=field_id,
            timestamp=datetime.now(timezone.utc),
            **bounded,
        )
