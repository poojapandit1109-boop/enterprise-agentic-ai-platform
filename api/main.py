
from fastapi import FastAPI

from src.config import get_settings

from api.routes.health import router as health_router
from api.routes.tasks import router as tasks_router


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description=(
        "Enterprise multi-agent AI platform "
        "for research, analytics, and decision support."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Register API routers
app.include_router(health_router)
app.include_router(tasks_router)


@app.get("/", tags=["General"])
def root():
    """Return basic application information."""

    return {
        "application": settings.app_name,
        "environment": settings.app_env,
        "version": "0.1.0",
        "status": "running",
        "documentation": "/docs",
    }
