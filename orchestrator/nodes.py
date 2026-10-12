# ""LangGraph nodes. Each node delegates a single domain responsibility."""

from agents.disease_agent.service import DiseaseAgent
from agents.field_agent.src.simulator_agent import FieldSimulatorAgent
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


def make_field_node(field_agent: FieldSimulatorAgent):
    """Create the optional field-data node for a selected field."""

    def field_node(state: AgriMindState) -> AgriMindState:
        field_id = state.get("field_id")
        if not field_id:
            return {
                "field_data": None,
                "agent_runs": [
                    {
                        "name": "field_agent",
                        "status": "skipped",
                        "warnings": ["Field data was skipped because no field_id was provided."],
                    }
                ],
                "warnings": ["Add a field_id to include soil context."],
            }

        try:
            result = field_agent.get_field_data(str(field_id))
        except (ValueError, RuntimeError) as error:
            return {
                "field_data": None,
                "agent_runs": [
                    {
                        "name": "field_agent",
                        "status": "failed",
                        "warnings": [str(error)],
                    }
                ],
                "warnings": [f"Field context is unavailable: {error}"],
                "status": "partial",
            }

        return {
            "field_data": result["data"],
            "agents_used": ["field_agent"],
            "agent_runs": [
                {
                    "name": "field_agent",
                    "status": "success",
                    "warnings": result["warnings"],
                }
            ],
            "warnings": result["warnings"],
        }

    return field_node