
from src.orchestration.state import AgentState


def supervisor_node(state: AgentState) -> dict:
    """
    Inspect the user task and select the next agent.

    Initial deterministic routing:
    - ResearchAgent for research and document-oriented questions.
    - SQLAnalystAgent for database and revenue questions.
    - PythonAnalystAgent for calculations and statistical analysis.
    """

    user_task = state.get("user_task", "").strip().lower()

    if not user_task:
        return {
            "current_agent": "SupervisorAgent",
            "next_agent": "SupervisorAgent",
            "workflow_status": "failed",
            "errors": ["User task cannot be empty."],
        }

    if any(
        keyword in user_task
        for keyword in ["sql", "database", "revenue", "sales", "query"]
    ):
        next_agent = "SQLAnalystAgent"

    elif any(
        keyword in user_task
        for keyword in ["calculate", "statistics", "correlation", "forecast"]
    ):
        next_agent = "PythonAnalystAgent"

    else:
        next_agent = "ResearchAgent"

    return {
        "current_agent": "SupervisorAgent",
        "next_agent": next_agent,
        "workflow_status": "running",
    }
