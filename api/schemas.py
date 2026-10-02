"""Public request and response contracts for the AgriMind API."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class DiseasePrediction(BaseModel):
    """A normalized result from the disease detection model."""

    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    model_version: str

class AgentExecution(BaseModel):
    """Traceable, safe-to-return summary of an agent invocation."""

    name: str
    status: Literal["success", "skipped", "failed"]
    warnings: list[str] = Field(default_factory=list)

class AnalyzeResponse(BaseModel):
    """Stable API response that can grow as new agents are connected."""
    status: Literal["success", "partial"]
    crop: str
    disease: DiseasePrediction | None = None
    weather: dict[str, Any] | None = None
    treatment: dict[str, Any] | None = None
    agents_used: list[str] = Field(default_factory=list)
    agent_runs: list[AgentExecution] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
