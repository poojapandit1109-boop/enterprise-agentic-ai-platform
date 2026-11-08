
from unittest.mock import MagicMock

import pytest

from src.agents.sql_validator import SQLValidationError
from src.tools.read_only_sql_tool import ReadOnlySQLTool


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.dialect.name = "postgresql"

    connection = engine.connect.return_value.__enter__.return_value

    result = connection.execute.return_value
    result.mappings.return_value.fetchmany.return_value = [
        {"id": 1},
        {"id": 2},
    ]

    return engine


@pytest.fixture
def sql_tool(mock_engine):
    return ReadOnlySQLTool(
        allowed_tables={"customers"},
        database_engine=mock_engine,
        max_rows=100,
        statement_timeout_ms=5000,
    )


def test_valid_select_executes_successfully(sql_tool, mock_engine):
    response = sql_tool.execute(
        "SELECT id FROM customers"
    )

    assert response["success"] is True
    assert response["is_read_only"] is True
    assert response["row_count"] == 2
    assert response["truncated"] is False
    assert response["tables_used"] == ["customers"]
    assert response["rows"] == [
        {"id": 1},
        {"id": 2},
    ]

    connection = mock_engine.connect.return_value.__enter__.return_value
    connection.execute.assert_called_once()


def test_database_transaction_is_read_only(sql_tool, mock_engine):
    sql_tool.execute("SELECT id FROM customers")

    connection = mock_engine.connect.return_value.__enter__.return_value

    connection.exec_driver_sql.assert_any_call(
        "SET TRANSACTION READ ONLY"
    )


def test_statement_timeout_is_applied(sql_tool, mock_engine):
    sql_tool.execute("SELECT id FROM customers")

    connection = mock_engine.connect.return_value.__enter__.return_value

    connection.exec_driver_sql.assert_any_call(
        "SET LOCAL statement_timeout = '5000ms'"
    )


def test_query_result_is_limited_to_max_rows(mock_engine):
    mock_engine.connect.return_value.__enter__.return_value.execute.return_value.mappings.return_value.fetchmany.return_value = [
        {"id": 1},
        {"id": 2},
        {"id": 3},
    ]

    tool = ReadOnlySQLTool(
        allowed_tables={"customers"},
        database_engine=mock_engine,
        max_rows=2,
    )

    response = tool.execute("SELECT id FROM customers")

    assert response["row_count"] == 2
    assert response["truncated"] is True
    assert response["rows"] == [
        {"id": 1},
        {"id": 2},
    ]


def test_delete_query_is_rejected_before_database_access(
    sql_tool,
    mock_engine,
):
    with pytest.raises(SQLValidationError):
        sql_tool.execute("DELETE FROM customers")

    mock_engine.connect.assert_not_called()


def test_update_query_is_rejected_before_database_access(
    sql_tool,
    mock_engine,
):
    with pytest.raises(SQLValidationError):
        sql_tool.execute(
            "UPDATE customers SET id = 1"
        )

    mock_engine.connect.assert_not_called()


def test_unapproved_table_is_rejected_before_database_access(
    sql_tool,
    mock_engine,
):
    with pytest.raises(SQLValidationError):
        sql_tool.execute("SELECT id FROM employees")

    mock_engine.connect.assert_not_called()


def test_multiple_statements_are_rejected(
    sql_tool,
    mock_engine,
):
    with pytest.raises(SQLValidationError):
        sql_tool.execute(
            "SELECT id FROM customers; SELECT id FROM customers"
        )

    mock_engine.connect.assert_not_called()


def test_query_parameters_are_passed_separately(
    sql_tool,
    mock_engine,
):
    sql_tool.execute(
        "SELECT id FROM customers WHERE id = :customer_id",
        parameters={"customer_id": 10},
    )

    connection = mock_engine.connect.return_value.__enter__.return_value

    connection.execute.assert_called_once()

    _, supplied_parameters = connection.execute.call_args.args

    assert supplied_parameters == {"customer_id": 10}


def test_invalid_parameters_are_rejected(
    sql_tool,
    mock_engine,
):
    with pytest.raises(TypeError):
        sql_tool.execute(
            "SELECT id FROM customers",
            parameters=["invalid"],
        )

    mock_engine.connect.assert_not_called()


def test_invalid_parameter_names_are_rejected(
    sql_tool,
    mock_engine,
):
    with pytest.raises(ValueError):
        sql_tool.execute(
            "SELECT id FROM customers",
            parameters={123: "invalid"},
        )

    mock_engine.connect.assert_not_called()


def test_non_postgresql_engine_is_rejected(mock_engine):
    mock_engine.dialect.name = "sqlite"

    tool = ReadOnlySQLTool(
        allowed_tables={"customers"},
        database_engine=mock_engine,
    )

    with pytest.raises(RuntimeError):
        tool.execute("SELECT id FROM customers")

    mock_engine.connect.assert_not_called()


@pytest.mark.parametrize(
    "max_rows",
    [0, -1, True, 1.5],
)
def test_invalid_max_rows_is_rejected(max_rows, mock_engine):
    with pytest.raises(ValueError):
        ReadOnlySQLTool(
            allowed_tables={"customers"},
            database_engine=mock_engine,
            max_rows=max_rows,
        )


@pytest.mark.parametrize(
    "timeout",
    [0, -1, True, 1.5],
)
def test_invalid_statement_timeout_is_rejected(
    timeout,
    mock_engine,
):
    with pytest.raises(ValueError):
        ReadOnlySQLTool(
            allowed_tables={"customers"},
            database_engine=mock_engine,
            statement_timeout_ms=timeout,
        )
