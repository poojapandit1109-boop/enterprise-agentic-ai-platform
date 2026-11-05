
import json
import re

from src.llm.base import BaseLLMClient, LLMResponse


class SQLMockLLMClient(BaseLLMClient):
    """
    Generate deterministic structured SQL output without API calls.

    This mock is for development and testing only.
    """

    def generate(self, prompt: str) -> LLMResponse:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        question_match = re.search(
            r"Business question:\s*(.*?)\n\nAvailable database schema:",
            prompt,
            re.DOTALL,
        )

        schema_match = re.search(
            r"Available database schema:\s*(.*?)\n\nInstructions:",
            prompt,
            re.DOTALL,
        )

        if not question_match or not schema_match:
            raise ValueError(
                "Business question or database schema could not be extracted."
            )

        business_question = question_match.group(1).strip()
        database_schema = schema_match.group(1).strip()

        if not business_question or not database_schema:
            raise ValueError(
                "Business question and database schema cannot be empty."
            )

        table_match = re.search(
            r"\b(?:CREATE\s+TABLE|TABLE)\s+"
            r"(?:IF\s+NOT\s+EXISTS\s+)?"
            r"(?:public\.)?([a-zA-Z_][a-zA-Z0-9_]*)",
            database_schema,
            re.IGNORECASE,
        )

        if not table_match:
            raise ValueError(
                "Mock SQL client requires a recognizable table definition."
            )

        table_name = table_match.group(1)

        response = {
            "agent_name": "SQLAnalystAgent",
            "status": "completed",
            "business_question": business_question,
            "sql_query": f"SELECT * FROM {table_name}",
            "explanation": (
                f"This simulated query retrieves records from "
                f"the {table_name} table. It is demonstration SQL "
                f"and has not been executed."
            ),
            "tables_used": [table_name],
            "is_read_only": True,
        }

        return LLMResponse(
            content=json.dumps(response),
            model="sql-mock-llm",
            is_mock=True,
        )
