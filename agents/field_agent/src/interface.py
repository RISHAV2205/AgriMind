"""Contracts for interchangeable field-data providers."""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class FieldReading:
    """One normalized snapshot of soil values for a field."""

    field_id: str
    soil_moisture_percent: float
    soil_temperature_c: float
    soil_ph: float
    nitrogen_mg_kg: float
    phosphorus_mg_kg: float
    potassium_mg_kg: float
    timestamp: datetime


class FieldDataSource(Protocol):
    """Any real sensor, database, or simulator must provide this operation."""

    def get_reading(self, field_id: str) -> FieldReading:
        """Return the latest normalized field reading for one field."""
