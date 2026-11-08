
"""Create the platform's default permission-controlled tool registry."""

from sqlalchemy import Engine

from src.tools.read_only_sql_tool import ReadOnlySQLTool
from src.tools.tool_registry import ToolRegistry


def build_tool_registry(
    allowed_tables: set[str] | list[str] | tuple[str, ...],
    database_engine: Engine | None = None,
    max_rows: int = 100,
    statement_timeout_ms: int = 5000,
) -> ToolRegistry:
    """Build a registry containing the authorized read-only SQL tool."""

    sql_tool = ReadOnlySQLTool(
        allowed_tables=allowed_tables,
        database_engine=database_engine,
        max_rows=max_rows,
        statement_timeout_ms=statement_timeout_ms,
    )

    registry = ToolRegistry()

    registry.register(
        "read_only_sql",
        sql_tool.execute,
    )

    return registry
