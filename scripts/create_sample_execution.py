
from uuid import UUID

from src.database.connection import SessionLocal
from src.database.repository import (
    get_task,
    create_agent_execution,
)

TASK_ID = UUID("f26780c9-3874-41b2-98db-b8852e280e4a")


def main():
    db = SessionLocal()

    try:
        task = get_task(db, TASK_ID)

        if task is None:
            print("ERROR: Task not found.")
            return

        execution = create_agent_execution(
            db=db,
            task_id=TASK_ID,
            agent_name="ResearchAgent",
            output=(
                "Revenue analysis research completed. "
                "Collected quarterly revenue trends "
                "and identified potential decline factors."
            ),
            metadata_json={
                "model": "gpt-4.1",
                "execution_type": "research",
                "source": "manual_test",
            },
        )

        print("Agent execution created successfully.")
        print("Execution ID:", execution.id)
        print("Task ID:", execution.task_id)
        print("Agent:", execution.agent_name)
        print("Status:", execution.status)

    except Exception as error:
        print("ERROR:", error)
        raise

    finally:
        db.close()


if __name__ == "__main__":
    main()
