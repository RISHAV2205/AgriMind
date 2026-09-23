"""The shared, explicit state passed between orchestration nodes."""

from typing import Any, Literal, TypedDict


class AgriMindState(TypedDict, total=False):
    crop: str
    image_bytes: bytes
    disease: dict[str, Any]
    weather: dict[str, Any] | None
    treatment: dict[str, Any] | None
    agents_used: list[str]
    agent_runs: list[dict[str, Any]]
    warnings: list[str]
    status: Literal["success", "partial"]
