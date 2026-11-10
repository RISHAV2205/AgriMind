"""LangGraph definition for the current AgriMind execution flow."""

from langgraph.graph import END, START, StateGraph

from agents.disease_agent.service import DiseaseAgent
from agents.field_agent.src.simulator_agent import FieldSimulatorAgent
from agents.weather_agent.service import WeatherAgent
from agents.recommendation_agent.service import RecommendationAgent
from orchestrator.nodes import (
    make_disease_node,
    make_field_node,
    make_recommendation_node,
    make_weather_node,
)
from orchestrator.state import AgriMindState


def build_graph(
    disease_agent: DiseaseAgent,
    weather_agent: WeatherAgent,
    field_agent: FieldSimulatorAgent | None = None,
    recommendation_agent: RecommendationAgent | None = None,
):
    """Build the graph without coupling it to HTTP or a particular model instance."""
    graph = StateGraph(AgriMindState)
    graph.add_node("disease_agent", make_disease_node(disease_agent))
    graph.add_node("weather_agent", make_weather_node(weather_agent))

    if field_agent is not None:
        graph.add_node("field_agent", make_field_node(field_agent))
    if recommendation_agent is not None:
        graph.add_node(
            "recommendation_agent",
            make_recommendation_node(recommendation_agent),
        )

    graph.add_edge(START, "disease_agent")
    graph.add_edge("disease_agent", "weather_agent")

    if field_agent is not None:
        graph.add_edge("weather_agent", "field_agent")
        next_node = "recommendation_agent" if recommendation_agent is not None else END
        graph.add_edge("field_agent", next_node)
    else:
        next_node = "recommendation_agent" if recommendation_agent is not None else END
        graph.add_edge("weather_agent", next_node)

    if recommendation_agent is not None:
        graph.add_edge("recommendation_agent", END)

    return graph.compile()
