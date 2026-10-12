"""Application-facing orchestrator; future routing belongs here, not in routes."""

from agents.disease_agent.service import DiseaseAgent
from agents.field_agent.src.simulator_agent import FieldSimulatorAgent
from agents.weather_agent.service import WeatherAgent
from orchestrator.graph import build_graph


class AgriMindOrchestrator:
    """Coordinates specialist agents through a LangGraph workflow."""

    def __init__(self) -> None:
        self._disease_agent = DiseaseAgent()
        self._weather_agent = WeatherAgent()
        self._field_agent = FieldSimulatorAgent()
        self._graph = build_graph(
            self._disease_agent,
            self._weather_agent,
            self._field_agent,
        )

    def analyze(
        self,
        *,
        crop: str,
        image_bytes: bytes,
        location: str | None = None,
        field_id: str | int | None = None,
    ) -> dict:
        """Run the currently available agents and return shared workflow state."""
        normalized_crop = crop.strip()
        if not normalized_crop:
            raise ValueError("Crop is required.")

        state = self._graph.invoke(
            {
                "crop": normalized_crop,
                "image_bytes": image_bytes,
                "location": location.strip() if location else None,
                "field_id": str(field_id) if field_id is not None else None,
                "agents_used": [],
                "agent_runs": [],
                "warnings": [],
            }
        )
        return dict(state)