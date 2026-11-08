
from unittest.mock import MagicMock

import pytest

from src.agents.sql_validator import SQLValidationError
from src.tools.registry_factory import build_tool_registry


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.dialect.name = "postgresql"

    connection = engine.connect.return_value.__enter__.return_value

    connection.execute.return_value.mappings.return_value.fetchmany.return_value = [
        {"customer_id": 101, "revenue": 5000}
    ]

    return engine


@pytest.fixture
def registry(mock_engine):
    return build_tool_registry(
        allowed_tables={"customers"},
        database_engine=mock_engine,
    )


def test_registry_registers_read_only_sql_tool(registry):
    assert list(registry.registered_tools) == ["read_only_sql"]


def test_sql_analyst_can_execute_read_only_query(
    registry,
    mock_engine,
):
    result = registry.execute(
        "SQLAnalystAgent",
        "read_only_sql",
        query="SELECT customer_id, revenue FROM customers",
    )

    assert result["success"] is True
    assert result["is_read_only"] is True
    assert result["row_count"] == 1

    mock_engine.connect.assert_called_once()


def test_research_agent_cannot_execute_sql(
    registry,
    mock_engine,
):
    with pytest.raises(PermissionError):
        registry.execute(
            "ResearchAgent",
            "read_only_sql",
            query="SELECT customer_id FROM customers",
        )

    mock_engine.connect.assert_not_called()


def test_unknown_agent_cannot_execute_sql(
    registry,
    mock_engine,
):
    with pytest.raises(PermissionError):
        registry.execute(
            "UnknownAgent",
            "read_only_sql",
            query="SELECT customer_id FROM customers",
        )

    mock_engine.connect.assert_not_called()


def test_destructive_query_is_rejected_through_registry(
    registry,
    mock_engine,
):
    with pytest.raises(SQLValidationError):
        registry.execute(
            "SQLAnalystAgent",
            "read_only_sql",
            query="DELETE FROM customers",
        )

    mock_engine.connect.assert_not_called()


def test_unapproved_table_is_rejected_through_registry(
    registry,
    mock_engine,
):
    with pytest.raises(SQLValidationError):
        registry.execute(
            "SQLAnalystAgent",
            "read_only_sql",
            query="SELECT id FROM employees",
        )

    mock_engine.connect.assert_not_called()


def test_registry_does_not_register_unapproved_tools(registry):
    assert registry.is_registered("delete_database") is False
    assert registry.is_registered("write_database") is False


def test_unauthorized_execution_does_not_call_database_execute(
    registry,
    mock_engine,
):
    with pytest.raises(PermissionError):
        registry.execute(
            "ReportGeneratorAgent",
            "read_only_sql",
            query="SELECT customer_id FROM customers",
        )

    connection = mock_engine.connect.return_value.__enter__.return_value

    connection.execute.assert_not_called()
