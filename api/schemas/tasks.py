
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class TaskCreate(BaseModel):
    title: str = Field(min_length=3, max_length=255)
    input_data: dict[str, Any] = Field(default_factory=dict)


class TaskStatusUpdate(BaseModel):
    status: Literal[
        "pending",
        "running",
        "completed",
        "failed",
    ]


class TaskResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    status: str
    input_data: dict[str, Any]
    created_at: datetime


class AgentExecutionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    task_id: UUID
    agent_name: str
    status: str
    output: str | None
    metadata_json: dict[str, Any]
    created_at: datetime
