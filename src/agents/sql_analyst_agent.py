
from src.agents.sql_mock_client import SQLMockLLMClient
from src.agents.sql_validator import (
    SQLValidationError,
    validate_read_only_sql,
)
from src.config import get_settings
from src.llm.base import BaseLLMClient
from src.llm.factory import get_reliable_llm_client
from src.llm.prompts import get_prompt
from src.llm.structured_client import generate_structured_output
from src.llm.structured_outputs import SQLAnalysisOutput
from src.orchestration.state import AgentResult


class SQLAnalystAgent:
    """Generate structured SQL and independently validate its safety."""

    def __init__(self, client: BaseLLMClient | None = None) -> None:
        if client is not None:
            self.client = client
        elif get_settings().llm_provider.strip().lower() == "mock":
            self.client = SQLMockLLMClient()
        else:
            self.client = get_reliable_llm_client()

        self.prompt_template = get_prompt("sql_analyst")

    def run(
        self,
        business_question: str,
        database_schema: str,
        allowed_tables: set[str] | list[str] | tuple[str, ...],
    ) -> AgentResult:
        """Generate and validate a read-only SQL query."""

        if (
            not isinstance(business_question, str)
            or not business_question.strip()
        ):
            raise ValueError("Business question cannot be empty.")

        if (
            not isinstance(database_schema, str)
            or not database_schema.strip()
        ):
            raise ValueError("Database schema cannot be empty.")

        if not allowed_tables:
            raise ValueError("Allowed tables cannot be empty.")

        prompt = self.prompt_template.render(
            business_question=business_question.strip(),
            database_schema=database_schema.strip(),
        )

        result = generate_structured_output(
            client=self.client,
            prompt=prompt,
            schema=SQLAnalysisOutput,
        )

        if not isinstance(result, SQLAnalysisOutput):
            raise TypeError(
                "SQL Analyst received an unexpected output schema."
            )

        if result.business_question != business_question.strip():
            raise ValueError(
                "Generated output does not match the original business question."
            )

        validation = validate_read_only_sql(
            query=result.sql_query,
            allowed_tables=allowed_tables,
        )

        reported_tables = {
            table.strip().lower()
            for table in result.tables_used
        }

        actual_tables = {
            table.lower()
            for table in validation.tables_used
        }

        if reported_tables != actual_tables:
            raise SQLValidationError(
                "Reported tables do not match the tables in the SQL query."
            )

        output = result.model_dump()

        # Use the independently validated table names.
        output["tables_used"] = list(validation.tables_used)

        # Return the normalized query produced by SQLGlot.
        output["sql_query"] = validation.normalized_query

        return {
            "agent_name": "SQLAnalystAgent",
            "status": result.status,
            "output": output,
            "error": None,
        }
