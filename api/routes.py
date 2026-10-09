"""Versioned HTTP routes. Routes validate HTTP concerns only."""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from api.schemas import (
    AgentExecution,
    AnalyzeResponse,
    DiseasePrediction,
    FarmerSummary,
    FieldDetailResponse,
    FieldReadingResponse,
    FieldSummary,
)
from agents.field_agent.src.farm_repository import FarmRepository
from agents.field_agent.src.repository import FieldReadingRepository
from agents.field_agent.src.simulator_agent import FieldSimulatorAgent
from orchestrator.agent import AgriMindOrchestrator

router = APIRouter(prefix="/api/v1", tags=["analysis"])
orchestrator = AgriMindOrchestrator()
field_simulator = FieldSimulatorAgent()

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
DEFAULT_CROP = "Tomato"


def _farm_repository() -> FarmRepository:
    """Build the repository lazily so importing routes never needs a database."""
    try:
        return FarmRepository()
    except (ValueError, RuntimeError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Farm data store is not configured.",
        ) from error


def _field_reading_repository() -> FieldReadingRepository:
    """Build the sensor-reading repository lazily as well."""
    try:
        return FieldReadingRepository()
    except (ValueError, RuntimeError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Field reading store is not configured.",
        ) from error


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_crop(
    crop: str = Form(..., min_length=2, max_length=80),
    image: UploadFile = File(...),
    location: str | None = Form(default=None, max_length=160),
) -> AnalyzeResponse:
    """Analyze a crop image and optionally add local weather context."""
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
        result = orchestrator.analyze(
            crop=crop,
            image_bytes=image_bytes,
            location=location,
        )

    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    # print(result)

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


@router.get("/farmers", response_model=list[FarmerSummary])
def list_farmers() -> list[FarmerSummary]:
    """List every farmer so the dashboard can offer a simple farmer picker."""
    repository = _farm_repository()
    return [FarmerSummary(**farmer) for farmer in repository.list_farmers()]


@router.get("/farmers/{farmer_id}/fields", response_model=list[FieldSummary])
def list_fields(farmer_id: int) -> list[FieldSummary]:
    """List the fields a farmer owns (Field 1, Field 2, ...)."""
    repository = _farm_repository()
    try:
        fields = repository.list_fields(farmer_id)
    except Exception as error:  # noqa: BLE001 - surfaced as a clean 503
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not load fields.",
        ) from error
    return [FieldSummary(**field) for field in fields]


@router.get("/fields/{field_id}", response_model=FieldDetailResponse)
def get_field_detail(
    field_id: int,
    location: str | None = None,
    crop: str | None = None,
) -> FieldDetailResponse:
    """Return one field with its latest soil reading and optional weather.

    The latest stored reading is used when available; otherwise a simulated
    reading is generated so the dashboard still has something to show during
    development.
    """
    repository = _farm_repository()
    field = repository.get_field(field_id)
    if field is None:
        raise HTTPException(status_code=404, detail="Field not found.")

    warnings: list[str] = []

    reading: FieldReadingResponse | None = None
    readings = _field_reading_repository()
    stored = readings.get_latest(str(field_id))
    if stored is not None:
        reading = FieldReadingResponse(
            field_id=stored.field_id,
            soil_moisture_percent=stored.soil_moisture_percent,
            soil_temperature_c=stored.soil_temperature_c,
            soil_ph=stored.soil_ph,
            nitrogen_mg_kg=stored.nitrogen_mg_kg,
            phosphorus_mg_kg=stored.phosphorus_mg_kg,
            potassium_mg_kg=stored.potassium_mg_kg,
            timestamp=stored.timestamp.isoformat(),
        )
    else:
        simulated = field_simulator.get_field_data(str(field_id))
        data = simulated["data"]
        reading = FieldReadingResponse(
            field_id=data["field_id"],
            soil_moisture_percent=data["soil_moisture_percent"],
            soil_temperature_c=data["soil_temperature_c"],
            soil_ph=data["soil_ph"],
            nitrogen_mg_kg=data["nitrogen_mg_kg"],
            phosphorus_mg_kg=data["phosphorus_mg_kg"],
            potassium_mg_kg=data["potassium_mg_kg"],
            timestamp=data["timestamp"],
        )
        warnings.append("Showing simulated sensor data; no stored reading found.")

    weather: dict | None = None
    if location:
        try:
            weather_result = orchestrator._weather_agent.get_forecast(location)  # noqa: SLF001
            weather = weather_result["data"]
        except (ValueError, RuntimeError) as error:
            warnings.append(f"Weather unavailable: {error}")

    return FieldDetailResponse(
        field=FieldSummary(**field),
        crop=crop or DEFAULT_CROP,
        location=location,
        reading=reading,
        weather=weather,
        warnings=warnings,
    )


@router.post("/fields/{field_id}/analyze", response_model=AnalyzeResponse)
async def analyze_field_leaf(
    field_id: int,
    image: UploadFile = File(...),
    crop: str = Form(default=DEFAULT_CROP, min_length=2, max_length=80),
    location: str | None = Form(default=None, max_length=160),
) -> AnalyzeResponse:
    """Diagnose a leaf photo for a specific field.

    This is the mobile-friendly entry point: the farmer picks a field, uploads
    a photo, and the disease agent (plus weather when a location is known) runs.
    """
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
        result = orchestrator.analyze(
            crop=crop,
            image_bytes=image_bytes,
            location=location,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    disease = result.get("disease")
    warnings = list(result.get("warnings", []))
    warnings.insert(0, f"Diagnosis for field #{field_id}.")

    return AnalyzeResponse(
        status=result["status"],
        crop=result["crop"],
        disease=DiseasePrediction(**disease) if disease else None,
        weather=result.get("weather"),
        treatment=result.get("treatment"),
        agents_used=result.get("agents_used", []),
        agent_runs=[AgentExecution(**run) for run in result.get("agent_runs", [])],
        warnings=warnings,
    )
