"""Permission-controlled SQL execution for the LangGraph workflow."""

from sqlalchemy.exc import SQLAlchemyError

from src.orchestration.state import AgentState
from src.tools.registry_factory import build_tool_registry

DEFAULT_SQL_ALLOWED_TABLES = {"tasks", "agent_executions"}


def sql_tool_execution_node(
    state: AgentState,
    tool_registry=None,
) -> dict:
    """Execute a verified SQL query through the permission-controlled registry."""

    existing_results = state.get("tool_results", [])

    def failure(message: str) -> dict:
        return {
            "current_agent": "SQLToolExecutionAgent",
            "tool_results": existing_results
            + [{"success": False, "error": message}],
            "workflow_status": "failed",
            "errors": state.get("errors", []) + [message],
            "final_response": (
                state.get("final_response", "")
                + "\n\nSQL execution failed. "
                "No SQL results are being reported."
            ),
        }

    # Execution must be explicitly enabled.
    if not state.get("enable_sql_execution", False):
        return failure("SQL execution is disabled.")

    # A SQL Analyst result must exist.
    agent_results = state.get("agent_results", [])

    if not agent_results:
        return failure("SQL execution requires a SQL Analyst result.")

    specialist_result = agent_results[-1]

    # Only the SQL Analyst may request SQL execution.
    if specialist_result.get("agent_name") != "SQLAnalystAgent":
        return failure(
            "Only SQLAnalystAgent can request read-only SQL execution."
        )

    if specialist_result.get("status") != "completed":
        return failure(
            "SQL Analyst result must be completed before execution."
        )

    # Independent verification must have passed.
    verification_result = state.get("verification_result")

    if not verification_result:
        return failure("SQL execution requires independent verification.")

    verification_output = verification_result.get("output", {})

    if not isinstance(verification_output, dict):
        return failure("SQL verification output is invalid.")

    if verification_output.get("status") != "passed":
        return failure("SQL execution requires passed verification.")

    if verification_output.get("verified_agent") != "SQLAnalystAgent":
        return failure(
            "Verification does not identify SQLAnalystAgent."
        )

    # Extract the SQL query from the verified specialist output.
    specialist_output = specialist_result.get("output", {})

    if not isinstance(specialist_output, dict):
        return failure("SQL Analyst output is invalid.")

    query = specialist_output.get("sql_query")

    if not isinstance(query, str) or not query.strip():
        return failure("SQL Analyst did not provide a valid query.")

    try:
        # Build the registry only when one was not injected.
        if tool_registry is None:
            allowed_tables = state.get(
                "allowed_tables",
                DEFAULT_SQL_ALLOWED_TABLES,
            )

            tool_registry = build_tool_registry(
                allowed_tables=set(allowed_tables)
            )

        execution_result = tool_registry.execute(
            agent_name="SQLAnalystAgent",
            tool_name="read_only_sql",
            query=query,
        )

        if not isinstance(execution_result, dict):
            raise TypeError("SQL tool returned an invalid result.")

        rows = execution_result.get("rows")
        row_count = execution_result.get("row_count")

        if execution_result.get("success") is not True:
            raise RuntimeError(
                "SQL tool did not confirm successful execution."
            )

        if execution_result.get("is_read_only") is not True:
            raise RuntimeError(
                "SQL tool did not confirm read-only execution."
            )

        if not isinstance(rows, list):
            raise TypeError("SQL tool returned invalid rows.")

        if not all(isinstance(row, dict) for row in rows):
            raise TypeError(
                "SQL tool returned an invalid row structure."
            )

        if type(row_count) is not int or row_count != len(rows):
            raise ValueError(
                "SQL tool returned an inconsistent row count."
            )

        if not isinstance(execution_result.get("truncated"), bool):
            raise TypeError(
                "SQL tool returned an invalid truncation flag."
            )

        return {
            "current_agent": "SQLToolExecutionAgent",
            "tool_results": existing_results + [execution_result],
            "workflow_status": "completed",
            "final_response": (
                state.get("final_response", "")
                + "\n\nSQL Tool Execution: "
                "Successfully executed the verified read-only query."
                + f"\nRows returned: {row_count}"
                + f"\nResults truncated: {execution_result['truncated']}"
            ),
        }

    except (
        PermissionError,
        LookupError,
        ValueError,
        TypeError,
        RuntimeError,
        ConnectionError,
        TimeoutError,
        SQLAlchemyError,
    ) as exc:
        return failure(
            f"SQL tool execution failed: {type(exc).__name__}"
        )


def route_after_sql_tool_execution(state: AgentState) -> str:
    """Continue to reporting only after successful SQL tool execution."""

    if state.get("workflow_status") != "completed":
        return "end"

    tool_results = state.get("tool_results", [])

    if not tool_results:
        return "end"

    if tool_results[-1].get("success") is not True:
        return "end"

    return "report_generator"