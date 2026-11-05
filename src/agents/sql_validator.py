
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError


class SQLValidationError(ValueError):
    """Raised when a SQL query violates read-only safety rules."""


@dataclass(frozen=True)
class SQLValidationResult:
    """Result of validating a read-only SQL query."""

    is_valid: bool
    normalized_query: str
    tables_used: tuple[str, ...]


# SQLGlot expression types that can perform or request
# operations outside ordinary read-only SELECT queries.
UNSAFE_EXPRESSION_NAMES = (
    "Insert",
    "Update",
    "Delete",
    "Create",
    "Drop",
    "Alter",
    "Merge",
    "TruncateTable",
    "Command",
    "Grant",
    "Revoke",
    "Copy",
    "Into",
    "Lock",
)


def _normalize_identifier(value: str) -> str:
    """Normalize an SQL identifier for case-insensitive comparison."""

    return value.strip().strip('"').strip("`").strip("[]").lower()


def _get_allowed_expression_types() -> tuple[type, ...]:
    """Resolve unsafe expression classes supported by SQLGlot."""

    expression_types = []

    for name in UNSAFE_EXPRESSION_NAMES:
        expression_type = getattr(exp, name, None)

        if expression_type is not None:
            expression_types.append(expression_type)

    return tuple(expression_types)


def _extract_table_name(table: exp.Table) -> str:
    """Return a normalized table name, including its schema when present."""

    table_name = _normalize_identifier(table.name)

    if not table_name:
        raise SQLValidationError(
            "Unable to determine a referenced table name."
        )

    schema_name = _normalize_identifier(table.db)
    catalog_name = _normalize_identifier(table.catalog)

    if catalog_name and schema_name:
        return f"{catalog_name}.{schema_name}.{table_name}"

    if schema_name:
        return f"{schema_name}.{table_name}"

    return table_name


def validate_read_only_sql(
    query: str,
    allowed_tables: set[str] | list[str] | tuple[str, ...],
    dialect: str = "postgres",
) -> SQLValidationResult:
    """
    Validate a single read-only SELECT query.

    Rules:
    - Exactly one SQL statement.
    - The root query must be SELECT.
    - Unsafe SQL expressions are rejected.
    - Only explicitly allowed tables can be referenced.
    - CTE names are not treated as physical database tables.
    """

    if not isinstance(query, str) or not query.strip():
        raise SQLValidationError("SQL query cannot be empty.")

    if not allowed_tables:
        raise SQLValidationError(
            "An explicit list of allowed tables is required."
        )

    normalized_allowed_tables = {
        _normalize_identifier(table)
        for table in allowed_tables
        if isinstance(table, str) and table.strip()
    }

    if not normalized_allowed_tables:
        raise SQLValidationError(
            "No valid allowed table names were provided."
        )

    try:
        statements = sqlglot.parse(query, read=dialect)
    except ParseError as exc:
        raise SQLValidationError(
            f"SQL parsing failed: {exc}"
        ) from exc

    if len(statements) != 1 or statements[0] is None:
        raise SQLValidationError(
            "Exactly one SQL statement is allowed."
        )

    statement = statements[0]

    if not isinstance(statement, exp.Select):
        raise SQLValidationError(
            "Only SELECT queries are allowed."
        )

    unsafe_types = _get_allowed_expression_types()

    for node in statement.walk():
        if unsafe_types and isinstance(node, unsafe_types):
            raise SQLValidationError(
                f"Unsafe SQL operation detected: {type(node).__name__}."
            )

    # CTE aliases represent temporary query results, not physical tables.
    cte_names = {
        _normalize_identifier(cte.alias)
        for cte in statement.find_all(exp.CTE)
        if cte.alias
    }

    tables_used = set()

    for table in statement.find_all(exp.Table):
        table_name = _extract_table_name(table)

        if table_name in cte_names:
            continue

        # A qualified table must be explicitly whitelisted by its
        # qualified name. A bare table name is accepted only when
        # the query itself does not specify a schema or catalog.
        if table_name not in normalized_allowed_tables:
            raise SQLValidationError(
                f"Table '{table_name}' is not in the allowed schema."
            )

        tables_used.add(table_name)

    if not tables_used:
        raise SQLValidationError(
            "The query must reference at least one allowed database table."
        )

    return SQLValidationResult(
        is_valid=True,
        normalized_query=statement.sql(dialect=dialect),
        tables_used=tuple(sorted(tables_used)),
    )
