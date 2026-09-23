"""AgriMind FastAPI application."""

from fastapi import FastAPI

from api.routes import router

app = FastAPI(
    title="AgriMind API",
    version="0.1.0",
    description="Multi-agent agricultural decision-support backend.",
)
app.include_router(router)


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    """Lightweight liveness endpoint; it deliberately does not load ML models."""
    return {"status": "ok", "service": "agrimind-api"}
