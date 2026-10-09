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


class FarmerSummary(BaseModel):
    """A farmer and the number of fields they own."""

    id: int
    name: str
    field_count: int


class FieldSummary(BaseModel):
    """One field belonging to a farmer."""

    id: int
    name: str
    farmer_id: int


class FieldReadingResponse(BaseModel):
    """Latest soil snapshot for a single field."""

    field_id: str
    soil_moisture_percent: float
    soil_temperature_c: float
    soil_ph: float
    nitrogen_mg_kg: float
    phosphorus_mg_kg: float
    potassium_mg_kg: float
    timestamp: str


class FieldDetailResponse(BaseModel):
    """Everything the dashboard needs to render one field."""

    field: FieldSummary
    crop: str = "Tomato"
    location: str | None = None
    reading: FieldReadingResponse | None = None
    weather: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
