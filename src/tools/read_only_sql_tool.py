
"""Secure, read-only SQL execution tool for SQLAnalystAgent."""

from typing import Any

from sqlalchemy import Engine, text

from src.agents.sql_validator import validate_read_only_sql
from src.database.connection import engine as default_engine


class ReadOnlySQLTool:
    """Validate and execute bounded read-only PostgreSQL queries."""

    def __init__(
        self,
        allowed_tables: set[str] | list[str] | tuple[str, ...],
        database_engine: Engine | None = None,
        max_rows: int = 100,
        statement_timeout_ms: int = 5000,
    ) -> None:
        if not allowed_tables:
            raise ValueError("An explicit table allowlist is required.")

        if (
            not isinstance(max_rows, int)
            or isinstance(max_rows, bool)
            or max_rows < 1
        ):
            raise ValueError("max_rows must be a positive integer.")

        if (
            not isinstance(statement_timeout_ms, int)
            or isinstance(statement_timeout_ms, bool)
            or statement_timeout_ms < 1
        ):
            raise ValueError(
                "statement_timeout_ms must be a positive integer."
            )

        self.allowed_tables = frozenset(allowed_tables)
        self.engine = database_engine or default_engine
        self.max_rows = max_rows
        self.statement_timeout_ms = statement_timeout_ms

    def execute(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate and execute one read-only SQL query."""

        if parameters is None:
            parameters = {}

        if not isinstance(parameters, dict):
            raise TypeError("SQL parameters must be a dictionary.")

        if not all(isinstance(key, str) for key in parameters):
            raise ValueError("SQL parameter names must be strings.")

        if self.engine.dialect.name != "postgresql":
            raise RuntimeError(
                "Read-only SQL execution is supported only on PostgreSQL."
            )

        validation = validate_read_only_sql(
            query=query,
            allowed_tables=set(self.allowed_tables),
            dialect="postgres",
        )

        # Open a transaction and mark it READ ONLY before executing SQL.
        # PostgreSQL enforces this independently of the SQL validator.
        with self.engine.connect() as connection, connection.begin():
            connection.exec_driver_sql(
                "SET TRANSACTION READ ONLY"
            )

            # Apply a database-side execution timeout.
            connection.exec_driver_sql(
                f"SET LOCAL statement_timeout = "
                f"'{self.statement_timeout_ms}ms'"
            )

            result = connection.execute(
                text(validation.normalized_query),
                parameters,
            )

            # Fetch at most max_rows + 1 to detect truncation.
            rows = result.mappings().fetchmany(self.max_rows + 1)

            truncated = len(rows) > self.max_rows
            rows = rows[: self.max_rows]

            return {
                "success": True,
                "is_read_only": True,
                "query": validation.normalized_query,
                "tables_used": list(validation.tables_used),
                "row_count": len(rows),
                "truncated": truncated,
                "rows": [dict(row) for row in rows],
            }
