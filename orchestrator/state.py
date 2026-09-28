"""The shared, explicit state passed between orchestration nodes."""

import operator
from typing import Annotated, Any, Literal, TypedDict


class AgriMindState(TypedDict, total=False):
    crop: str
    image_bytes: bytes
    location: str | None
    disease: dict[str, Any]
    weather: dict[str, Any] | None
    treatment: dict[str, Any] | None
    # These fields collect outputs from every node instead of being overwritten.
    agents_used: Annotated[list[str], operator.add]
    agent_runs: Annotated[list[dict[str, Any]], operator.add]
    warnings: Annotated[list[str], operator.add]
    status: Literal["success", "partial"]
