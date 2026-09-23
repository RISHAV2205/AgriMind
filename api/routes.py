"""Versioned HTTP routes. Routes validate HTTP concerns only."""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from api.schemas import AnalyzeResponse, DiseasePrediction, AgentExecution
from orchestrator.agent import AgriMindOrchestrator

router = APIRouter(prefix="/api/v1", tags=["analysis"])
orchestrator = AgriMindOrchestrator()

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_crop(
    crop: str = Form(..., min_length=2, max_length=80),
    image: UploadFile = File(...),
) -> AnalyzeResponse:
    """Analyze a crop image through the multi-agent orchestrator."""
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Use a JPEG, PNG, or WebP image.",
        )

    image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")

    try:
        result = orchestrator.analyze(crop=crop, image_bytes=image_bytes)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    disease = result.get("disease")
    return AnalyzeResponse(
        status=result["status"],
        crop=result["crop"],
        disease=DiseasePrediction(**disease) if disease else None,
        weather=result.get("weather"),
        treatment=result.get("treatment"),
        agents_used=result.get("agents_used", []),
        agent_runs=[AgentExecution(**run) for run in result.get("agent_runs", [])],
        warnings=result.get("warnings", []),
    )
