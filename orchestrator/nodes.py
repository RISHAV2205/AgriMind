"""LangGraph nodes. Each node delegates a single domain responsibility."""

from agents.disease_agent.service import DiseaseAgent
from agents.weather_agent.service import WeatherAgent
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


def make_weather_node(weather_agent: WeatherAgent):
    """Create the optional weather node, keeping provider failures non-fatal."""

    def weather_node(state: AgriMindState) -> AgriMindState:
        location = state.get("location")
        if not location:
            return {
                "weather": None,
                "agent_runs": [
                    {
                        "name": "weather_agent",
                        "status": "skipped",
                        "warnings": ["Weather was skipped because no location was provided."],
                    }
                ],
                "warnings": ["Add a location to include weather context."],
            }

        try:
            result = weather_agent.get_forecast(location)
        except (ValueError, RuntimeError) as error:
            return {
                "weather": None,
                "agent_runs": [
                    {"name": "weather_agent", "status": "failed", "warnings": [str(error)]}
                ],
                "warnings": [f"Weather context is unavailable: {error}"],
                "status": "partial",
            }

        return {
            "weather": result["data"],
            "agents_used": ["weather_agent"],
            "agent_runs": [
                {
                    "name": "weather_agent",
                    "status": "success",
                    "warnings": result["warnings"],
                }
            ],
            "warnings": result["warnings"],
        }

    return weather_node
