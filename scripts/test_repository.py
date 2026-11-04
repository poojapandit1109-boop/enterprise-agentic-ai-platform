
from src.database.connection import SessionLocal
from src.database.repository import (
    create_agent_execution,
    create_task,
    delete_task,
    get_agent_executions,
    get_task,
    update_task_status,
)


def main():
    db = SessionLocal()
    task = None

    try:
        # 1. Create a task
        task = create_task(
            db=db,
            title="Analyze quarterly revenue decline",
            input_data={
                "department": "Finance",
                "quarter": "Q3",
            },
        )
        print("Task created:", task.id)

        # 2. Retrieve the task
        retrieved = get_task(db, task.id)
        assert retrieved is not None
        print("Task retrieved:", retrieved.title)

        # 3. Create an agent execution
        execution = create_agent_execution(
            db=db,
            task_id=task.id,
            agent_name="ResearchAgent",
            output="Revenue declined due to lower sales.",
            metadata_json={"source": "test"},
        )
        print("Agent execution created:", execution.id)

        # 4. Retrieve agent executions
        executions = get_agent_executions(db, task.id)
        assert len(executions) == 1
        print("Executions retrieved:", len(executions))

        # 5. Update task status
        updated = update_task_status(
            db=db,
            task_id=task.id,
            status="completed",
        )
        assert updated is not None
        print("Updated task status:", updated.status)

        # 6. Delete test task and verify cleanup
        task_id = task.id
        deleted = delete_task(db, task_id)
        assert deleted is True
        assert get_task(db, task_id) is None

        print("Task and associated executions deleted.")
        print("ALL REPOSITORY TESTS PASSED")

    finally:
        # Safety cleanup if an earlier test fails
        if task is not None:
            delete_task(db, task.id)
        db.close()


if __name__ == "__main__":
    main()
