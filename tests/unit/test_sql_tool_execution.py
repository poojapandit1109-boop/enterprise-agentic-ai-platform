
"""Integration tests for permission-controlled SQL tool execution."""

import pytest

from src.orchestration.graph import (
    route_after_verification,
    sql_tool_execution_node,
)
from src.tools.tool_registry import ToolRegistry


class FakeToolRegistry:
    """Fake registry that avoids connecting to PostgreSQL."""

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def execute(self, agent_name, tool_name, **kwargs):
        self.calls.append(
            {
                "agent_name": agent_name,
                "tool_name": tool_name,
                **kwargs,
            }
        )

        if self.error:
            raise self.error

        return self.result


def make_verified_sql_state():
    """Create a SQL result with successful independent verification."""

    return {
        "user_task": "Show the latest task records.",
        "workflow_status": "completed",
        "enable_sql_execution": True,
        "allowed_tables": ["tasks", "agent_executions"],
        "agent_results": [
            {
                "agent_name": "SQLAnalystAgent",
                "status": "completed",
                "output": {
                    "business_question": "Show the latest task records.",
                    "sql_query": "SELECT * FROM tasks",
                    "explanation": "Retrieve task records.",
                    "tables_used": ["tasks"],
                    "is_read_only": True,
                },
                "error": None,
            }
        ],
        "verification_result": {
            "agent_name": "VerificationAgent",
            "status": "completed",
            "output": {
                "agent_name": "VerificationAgent",
                "status": "passed",
                "verified_agent": "SQLAnalystAgent",
                "checks": [],
                "summary": "SQL verification passed.",
                "warnings": [],
            },
            "error": None,
        },
        "tool_results": [],
        "errors": [],
        "final_response": "SQL query verified.",
    }


def make_successful_tool_result():
    """Return a deterministic fake SQL execution result."""

    return {
        "success": True,
        "is_read_only": True,
        "query": "SELECT * FROM tasks",
        "tables_used": ["tasks"],
        "row_count": 2,
        "truncated": False,
        "rows": [
            {
                "id": "task-001",
                "title": "Research task",
                "status": "completed",
            },
            {
                "id": "task-002",
                "title": "Analysis task",
                "status": "pending",
            },
        ],
    }


def test_sql_execution_routes_only_after_passed_verification():
    state = make_verified_sql_state()

    assert route_after_verification(state) == "sql_tool_execution"


def test_sql_execution_is_disabled_by_default():
    state = make_verified_sql_state()
    state.pop("enable_sql_execution")

    assert route_after_verification(state) == "report_generator"


def test_sql_execution_requires_successful_verification():
    state = make_verified_sql_state()
    state["verification_result"]["output"]["status"] = "needs_review"
    state["workflow_status"] = "needs_review"

    assert route_after_verification(state) == "report_generator"


def test_sql_execution_uses_registry_and_stores_results():
    state = make_verified_sql_state()

    fake_registry = FakeToolRegistry(
        result=make_successful_tool_result()
    )

    result = sql_tool_execution_node(
        state,
        tool_registry=fake_registry,
    )

    assert result["workflow_status"] == "completed"

    assert len(result["tool_results"]) == 1

    stored_result = result["tool_results"][0]

    assert stored_result["success"] is True
    assert stored_result["is_read_only"] is True
    assert stored_result["row_count"] == 2
    assert len(stored_result["rows"]) == 2

    assert len(fake_registry.calls) == 1

    assert fake_registry.calls[0]["agent_name"] == "SQLAnalystAgent"
    assert fake_registry.calls[0]["tool_name"] == "read_only_sql"
    assert fake_registry.calls[0]["query"] == "SELECT * FROM tasks"


def test_sql_execution_failure_does_not_report_successful_results():
    state = make_verified_sql_state()

    fake_registry = FakeToolRegistry(
        error=RuntimeError("Simulated database failure.")
    )

    result = sql_tool_execution_node(
        state,
        tool_registry=fake_registry,
    )

    assert result["workflow_status"] == "failed"
    assert result["tool_results"][0]["success"] is False
    assert "RuntimeError" in result["tool_results"][0]["error"]
    assert "No SQL results are being reported." in result["final_response"]


def test_sql_execution_rejects_non_sql_specialist():
    state = make_verified_sql_state()
    state["agent_results"][0]["agent_name"] = "ResearchAgent"

    fake_registry = FakeToolRegistry(
        result=make_successful_tool_result()
    )

    result = sql_tool_execution_node(
        state,
        tool_registry=fake_registry,
    )

    assert result["workflow_status"] == "failed"
    assert fake_registry.calls == []


def test_sql_execution_rejects_unverified_query():
    state = make_verified_sql_state()
    state["verification_result"]["output"]["verified_agent"] = (
        "ResearchAgent"
    )

    fake_registry = FakeToolRegistry(
        result=make_successful_tool_result()
    )

    result = sql_tool_execution_node(
        state,
        tool_registry=fake_registry,
    )

    assert result["workflow_status"] == "failed"
    assert fake_registry.calls == []


def test_tool_registry_denies_unauthorized_agent():
    registry = ToolRegistry()

    registry.register(
        "read_only_sql",
        lambda **kwargs: {"success": True},
    )

    with pytest.raises(PermissionError):
        registry.execute(
            agent_name="ResearchAgent",
            tool_name="read_only_sql",
            query="SELECT * FROM tasks",
            parameters={},
        )
