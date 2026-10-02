"""Agent adapter exposing simulated field readings to the orchestrator."""

from agents.field_agent.data.simulator import FieldDataSimulator
from agents.field_agent.src.interface import FieldDataSource


class FieldSimulatorAgent:
    """Expose a FieldDataSource through AgriMind's common agent-result shape."""

    name = "field_simulator_agent"

    def __init__(self, source: FieldDataSource | None = None) -> None:
        self._source = source or FieldDataSimulator()

    def get_field_data(self, field_id: str) -> dict:
        """Return one simulated sensor snapshot for the specified field."""
        reading = self._source.get_reading(field_id)
        return {
            "agent": self.name,
            "status": "success",
            "data": {
                "field_id": reading.field_id,
                "soil_moisture_percent": reading.soil_moisture_percent,
                "soil_temperature_c": reading.soil_temperature_c,
                "soil_ph": reading.soil_ph,
                "nitrogen_mg_kg": reading.nitrogen_mg_kg,
                "phosphorus_mg_kg": reading.phosphorus_mg_kg,
                "potassium_mg_kg": reading.potassium_mg_kg,
                "timestamp": reading.timestamp.isoformat(),
                "source": "simulator",
            },
            "warnings": [],
        }
