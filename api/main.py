"""AgriMind FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI(
    title="AgriMind API",
    version="0.1.0",
    description="Multi-agent agricultural decision-support backend.",
)
app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    """Lightweight liveness endpoint; it deliberately does not load ML models."""
    return {"status": "ok", "service": "agrimind-api"}
