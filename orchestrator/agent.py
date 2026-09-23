"""Application-facing orchestrator; future routing belongs here, not in routes."""

from agents.disease_agent.service import DiseaseAgent
from orchestrator.graph import build_graph


class AgriMindOrchestrator:
    """Coordinates specialist agents through a LangGraph workflow."""

    def __init__(self) -> None:
        self._disease_agent = DiseaseAgent()
        self._graph = build_graph(self._disease_agent)

    def analyze(self, *, crop: str, image_bytes: bytes) -> dict:
        """Run the currently available agents and return shared workflow state."""
        normalized_crop = crop.strip()
        if not normalized_crop:
            raise ValueError("Crop is required.")

        state = self._graph.invoke(
            {
                "crop": normalized_crop,
                "image_bytes": image_bytes,
                "agents_used": [],
                "agent_runs": [],
                "warnings": [],
            }
        )
        return dict(state)
