
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models import AgentExecution, Task


def create_task(
    db: Session,
    title: str,
    input_data: dict,
) -> Task:
    """Create and persist a new task."""

    task = Task(
        title=title,
        input_data=input_data,
        status="pending",
    )

    try:
        db.add(task)
        db.commit()
        db.refresh(task)
        return task
    except Exception:
        db.rollback()
        raise


def get_task(db: Session, task_id: UUID) -> Task | None:
    """Retrieve a task by its UUID."""

    statement = select(Task).where(Task.id == task_id)
    return db.execute(statement).scalar_one_or_none()


def update_task_status(
    db: Session,
    task_id: UUID,
    status: str,
) -> Task | None:
    """Update the status of an existing task."""

    task = get_task(db, task_id)

    if task is None:
        return None

    try:
        task.status = status
        db.commit()
        db.refresh(task)
        return task
    except Exception:
        db.rollback()
        raise


def create_agent_execution(
    db: Session,
    task_id: UUID,
    agent_name: str,
    output: str | None = None,
    metadata_json: dict | None = None,
) -> AgentExecution:
    """Save an agent execution result."""

    execution = AgentExecution(
        task_id=task_id,
        agent_name=agent_name,
        status="completed",
        output=output,
        metadata_json=metadata_json or {},
    )

    try:
        db.add(execution)
        db.commit()
        db.refresh(execution)
        return execution
    except Exception:
        db.rollback()
        raise


def get_agent_executions(
    db: Session,
    task_id: UUID,
) -> list[AgentExecution]:
    """Retrieve all executions belonging to a task."""

    statement = (
        select(AgentExecution)
        .where(AgentExecution.task_id == task_id)
        .order_by(AgentExecution.created_at)
    )

    return list(db.execute(statement).scalars().all())


def delete_task(db: Session, task_id: UUID) -> bool:
    """Delete a task and its associated executions."""

    task = get_task(db, task_id)

    if task is None:
        return False

    try:
        db.delete(task)
        db.commit()
        return True
    except Exception:
        db.rollback()
        raise
