
from typing import Any, Literal, TypedDict


class AgentResult(TypedDict):
    agent_name: str
    status: Literal["completed", "failed", "needs_review"]
    output: dict[str, Any]
    error: str | None


class AgentState(TypedDict, total=False):
    """Shared state passed between agents in the LangGraph workflow."""

    task_id: str
    user_task: str

    # Database context supplied to the SQL Analyst Agent.
    database_schema: str
    allowed_tables: list[str]

    current_agent: str
    next_agent: str

    agent_results: list[AgentResult]

    workflow_status: Literal[
        "pending",
        "running",
        "completed",
        "failed",
        "needs_review",
    ]

    errors: list[str]

    final_response: str
