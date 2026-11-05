
import pytest

from src.agents.sql_validator import (
    SQLValidationError,
    validate_read_only_sql,
)


def test_accepts_valid_select_query():
    result = validate_read_only_sql(
        "SELECT id, name FROM customers",
        {"customers"},
    )

    assert result.is_valid is True
    assert result.tables_used == ("customers",)


def test_accepts_select_with_join():
    result = validate_read_only_sql(
        """
        SELECT c.id, o.total
        FROM customers c
        JOIN orders o ON c.id = o.customer_id
        """,
        {"customers", "orders"},
    )

    assert result.is_valid is True
    assert set(result.tables_used) == {"customers", "orders"}


def test_accepts_select_with_cte():
    result = validate_read_only_sql(
        """
        WITH recent_customers AS (
            SELECT id FROM customers
        )
        SELECT id FROM recent_customers
        """,
        {"customers"},
    )

    assert result.is_valid is True
    assert result.tables_used == ("customers",)


@pytest.mark.parametrize(
    "query",
    [
        "INSERT INTO customers (name) VALUES ('Test')",
        "UPDATE customers SET name = 'Test'",
        "DELETE FROM customers",
        "DROP TABLE customers",
        "ALTER TABLE customers ADD COLUMN age INT",
        "TRUNCATE TABLE customers",
        "CREATE TABLE test_table (id INT)",
    ],
)
def test_rejects_unsafe_statements(query):
    with pytest.raises(SQLValidationError):
        validate_read_only_sql(query, {"customers"})


def test_rejects_multiple_statements():
    query = (
        "SELECT id FROM customers; "
        "DELETE FROM customers"
    )

    with pytest.raises(
        SQLValidationError,
        match="Exactly one SQL statement",
    ):
        validate_read_only_sql(query, {"customers"})


def test_rejects_unapproved_table():
    with pytest.raises(
        SQLValidationError,
        match="not in the allowed schema",
    ):
        validate_read_only_sql(
            "SELECT * FROM employee_salaries",
            {"customers"},
        )


def test_rejects_empty_query():
    with pytest.raises(
        SQLValidationError,
        match="cannot be empty",
    ):
        validate_read_only_sql("", {"customers"})


def test_rejects_missing_allowed_tables():
    with pytest.raises(
        SQLValidationError,
        match="explicit list",
    ):
        validate_read_only_sql(
            "SELECT * FROM customers",
            set(),
        )


def test_rejects_malformed_sql():
    with pytest.raises(SQLValidationError):
        validate_read_only_sql(
            "SELECT FROM WHERE",
            {"customers"},
        )


def test_rejects_select_without_table():
    with pytest.raises(
        SQLValidationError,
        match="at least one allowed database table",
    ):
        validate_read_only_sql(
            "SELECT 1",
            {"customers"},
        )


def test_accepts_schema_qualified_table():
    result = validate_read_only_sql(
        "SELECT id FROM public.customers",
        {"public.customers"},
    )

    assert result.is_valid is True
    assert result.tables_used == ("public.customers",)
