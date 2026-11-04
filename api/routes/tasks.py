
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from src.database.connection import get_db
from src.database.repository import (
    create_task,
    get_task,
    update_task_status,
    create_agent_execution,
    get_agent_executions,
    delete_task,
)

from api.schemas.tasks import (
    TaskCreate,
    TaskStatusUpdate,
    TaskResponse,
    AgentExecutionResponse,
)

router = APIRouter(
    prefix="/api/v1/tasks",
    tags=["Tasks"],
)


@router.post(
    "",
    response_model=TaskResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_new_task(
    payload: TaskCreate,
    db: Session = Depends(get_db),
):
    """Create a new task."""

    return create_task(
        db=db,
        title=payload.title,
        input_data=payload.input_data,
    )


@router.get(
    "/{task_id}",
    response_model=TaskResponse,
)
def retrieve_task(
    task_id: UUID,
    db: Session = Depends(get_db),
):
    """Retrieve a task by ID."""

    task = get_task(db, task_id)

    if task is None:
        raise HTTPException(
            status_code=404,
            detail="Task not found",
        )

    return task


@router.patch(
    "/{task_id}/status",
    response_model=TaskResponse,
)
def change_task_status(
    task_id: UUID,
    payload: TaskStatusUpdate,
    db: Session = Depends(get_db),
):
    """Update task status."""

    task = update_task_status(
        db=db,
        task_id=task_id,
        status=payload.status,
    )

    if task is None:
        raise HTTPException(
            status_code=404,
            detail="Task not found",
        )

    return task


@router.get(
    "/{task_id}/executions",
    response_model=list[AgentExecutionResponse],
)
def retrieve_task_executions(
    task_id: UUID,
    db: Session = Depends(get_db),
):
    """Retrieve agent executions for a task."""

    task = get_task(db, task_id)

    if task is None:
        raise HTTPException(
            status_code=404,
            detail="Task not found",
        )

    return get_agent_executions(db, task_id)


@router.delete(
    "/{task_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def remove_task(
    task_id: UUID,
    db: Session = Depends(get_db),
):
    """Delete a task and associated executions."""

    deleted = delete_task(db, task_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Task not found",
        )

    return Response(status_code=204)
