from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.config import get_settings
from src.database.connection import check_database_connection
from src.database.redis_client import check_redis_connection
from api.schemas.health import (
    LivenessResponse,
    ReadinessResponse,
)

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/live", response_model=LivenessResponse)
def liveness():
    """Check whether the API process is running."""

    return {"status": "ok"}


@router.get("/ready", response_model=ReadinessResponse)
def readiness():
    """Check whether required infrastructure is available."""

    settings = get_settings()

    dependencies = {
        "postgresql": check_database_connection(),
        "redis": check_redis_connection(),
    }

    is_ready = all(dependencies.values())

    response = ReadinessResponse(
        status="ready" if is_ready else "not_ready",
        app_name=settings.app_name,
        dependencies=dependencies,
    )

    if not is_ready:
        return JSONResponse(
            status_code=503,
            content=response.model_dump(),
        )

    return response