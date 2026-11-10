
"""AgriMind FastAPI application."""

from contextlib import asynccontextmanager

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from agents.field_agent.src.reading_scheduler import generate_readings


scheduler = BackgroundScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting AgriMind API...", flush=True)

    scheduler.add_job(
        generate_readings,
        trigger="interval",
        seconds=10000,
        id="field_readings",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    scheduler.start()
    print("AgriMind reading scheduler started.", flush=True)

    try:
        yield
    finally:
        if scheduler.running:
            scheduler.shutdown(wait=True)
        print("AgriMind reading scheduler stopped.", flush=True)


app = FastAPI(
    title="AgriMind API",
    version="0.1.0",
    description="Multi-agent agricultural decision-support backend.",
    lifespan=lifespan,
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
    # • your frontend application can safely pass along HTTP cookies, Authorization headers (like Bearer tokens), and TLS client certificates. If you turn this to False, the browser will automatically strip these credentials out before hitting your endpoint, which breaks authentication systems like sessions or JWTs stored in cookies.
    allow_credentials=True,
    # The "*" wildcard acts as a universal green light for HTTP request types. It means your backend accepts any request format—whether it's fetching data (GET), submitting forms (POST), updating database fields (PUT, PATCH), or removing entries (DELETE).
    allow_methods=["*"],
    # This wildcard allows the frontend to include any custom or standard HTTP headers. This is highly important for modern single-page applications (SPAs), which regularly attach headers like Content-Type: application/json or custom authentication markers (e.g., X-API-KEY).
    allow_headers=["*"],
)


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    """Lightweight liveness endpoint."""
    return {"status": "ok", "service": "agrimind-api"}
