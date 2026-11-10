
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

    # Database context for SQL Analyst.
    database_schema: str
    allowed_tables: list[str]

    # Explicit opt-in for SQL execution.
    # Defaults to disabled when not supplied.
    enable_sql_execution: bool

    # Dataset context for Python Analyst.
    analysis_records: list[dict[str, Any]]
    analysis_objective: str

    # Agent routing.
    current_agent: str
    next_agent: str

    # Specialist outputs and independent verification.
    agent_results: list[AgentResult]
    verification_result: AgentResult

    # Results returned by permission-controlled tools.
    tool_results: list[dict[str, Any]]

    # Final executive report.
    report_result: AgentResult

    # Workflow tracking.
    workflow_status: Literal[
        "pending",
        "running",
        "completed",
        "failed",
        "needs_review",
    ]

    errors: list[str]
    final_response: str
