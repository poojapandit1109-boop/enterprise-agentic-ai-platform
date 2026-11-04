
import pytest
from uuid import uuid4

from src.database.connection import SessionLocal
from src.database.repository import (
    create_agent_execution,
    create_task,
    delete_task,
    get_agent_executions,
    get_task,
    update_task_status,
)


@pytest.fixture
def db():
    """Provide a database session for each test."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def task(db):
    """Create a temporary task and clean it up after testing."""
    new_task = create_task(
        db=db,
        title="Automated repository test",
        input_data={"source": "pytest"},
    )

    yield new_task

    delete_task(db, new_task.id)


def test_create_and_retrieve_task(db, task):
    retrieved = get_task(db, task.id)

    assert retrieved is not None
    assert retrieved.id == task.id
    assert retrieved.title == "Automated repository test"
    assert retrieved.status == "pending"


def test_update_task_status(db, task):
    updated = update_task_status(
        db=db,
        task_id=task.id,
        status="completed",
    )

    assert updated is not None
    assert updated.status == "completed"


def test_create_and_retrieve_agent_execution(db, task):
    create_agent_execution(
        db=db,
        task_id=task.id,
        agent_name="ResearchAgent",
        output="Research completed successfully.",
        metadata_json={"source": "pytest"},
    )

    executions = get_agent_executions(db, task.id)

    assert len(executions) == 1
    assert executions[0].agent_name == "ResearchAgent"
    assert executions[0].status == "completed"


def test_delete_task_cascades_to_executions(db, task):
    create_agent_execution(
        db=db,
        task_id=task.id,
        agent_name="ResearchAgent",
        output="Temporary result",
    )

    task_id = task.id

    assert delete_task(db, task_id) is True
    assert get_task(db, task_id) is None
    assert get_agent_executions(db, task_id) == []


def test_get_nonexistent_task(db):
    result = get_task(db, uuid4())

    assert result is None


def test_update_nonexistent_task(db):
    result = update_task_status(
        db=db,
        task_id=uuid4(),
        status="completed",
    )

    assert result is None
