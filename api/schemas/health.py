from typing import Literal

from pydantic import BaseModel


class LivenessResponse(BaseModel):
    status: Literal["ok"]


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    app_name: str
    dependencies: dict[str, bool]