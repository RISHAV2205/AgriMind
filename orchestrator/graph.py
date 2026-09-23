"""LangGraph definition for the current AgriMind execution flow."""

from langgraph.graph import END, START, StateGraph

from agents.disease_agent.service import DiseaseAgent
from orchestrator.nodes import make_disease_node
from orchestrator.state import AgriMindState


def build_graph(disease_agent: DiseaseAgent):
    """Build the graph without coupling it to HTTP or a particular model instance."""
    graph = StateGraph(AgriMindState)
    graph.add_node("disease_agent", make_disease_node(disease_agent))
    graph.add_edge(START, "disease_agent")
    graph.add_edge("disease_agent", END)
    return graph.compile()
