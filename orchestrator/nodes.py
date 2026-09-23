"""LangGraph nodes. Each node delegates a single domain responsibility."""

from agents.disease_agent.service import DiseaseAgent
from orchestrator.state import AgriMindState


def make_disease_node(disease_agent: DiseaseAgent):
    """Create a graph node with the disease agent injected explicitly."""

    def disease_node(state: AgriMindState) -> AgriMindState:
        result = disease_agent.analyze_image(state["image_bytes"])
        return {
            "disease": result["data"],
            "agents_used": ["disease_agent"],
            "agent_runs": [
                {
                    "name": "disease_agent",
                    "status": "success",
                    "warnings": result["warnings"],
                }
            ],
            "warnings": result["warnings"],
            "status": "success",
        }

    return disease_node
