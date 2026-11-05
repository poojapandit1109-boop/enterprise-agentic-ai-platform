
import json

import pytest

from src.agents.sql_analyst_agent import SQLAnalystAgent
from src.agents.sql_mock_client import SQLMockLLMClient
from src.agents.sql_validator import SQLValidationError
from src.llm.base import BaseLLMClient, LLMResponse


DATABASE_SCHEMA = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100),
    city VARCHAR(100)
);
"""


class FakeSQLLLMClient(BaseLLMClient):
    """Return controlled structured SQL responses for testing."""

    def __init__(self, response: dict):
        self.response = response

    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(
            content=json.dumps(self.response),
            model="fake-sql-llm",
            is_mock=True,
        )


def valid_response() -> dict:
    return {
        "agent_name": "SQLAnalystAgent",
        "status": "completed",
        "business_question": "Find all customers",
        "sql_query": "SELECT * FROM customers",
        "explanation": "Retrieve all customer records.",
        "tables_used": ["customers"],
        "is_read_only": True,
    }


def test_sql_agent_returns_validated_result():
    agent = SQLAnalystAgent()

    result = agent.run(
        business_question="Find all customers",
        database_schema=DATABASE_SCHEMA,
        allowed_tables={"customers"},
    )

    assert result["agent_name"] == "SQLAnalystAgent"
    assert result["status"] == "completed"
    assert result["error"] is None
    assert result["output"]["tables_used"] == ["customers"]
    assert result["output"]["is_read_only"] is True
    assert result["output"]["sql_query"] == "SELECT * FROM customers"


def test_sql_agent_uses_structured_mock_client():
    agent = SQLAnalystAgent()

    assert isinstance(agent.client, SQLMockLLMClient)


def test_sql_agent_rejects_empty_business_question():
    agent = SQLAnalystAgent()

    with pytest.raises(ValueError, match="Business question"):
        agent.run(
            business_question=" ",
            database_schema=DATABASE_SCHEMA,
            allowed_tables={"customers"},
        )


def test_sql_agent_rejects_empty_database_schema():
    agent = SQLAnalystAgent()

    with pytest.raises(ValueError, match="Database schema"):
        agent.run(
            business_question="Find customers",
            database_schema=" ",
            allowed_tables={"customers"},
        )


def test_sql_agent_rejects_empty_allowed_tables():
    agent = SQLAnalystAgent()

    with pytest.raises(ValueError, match="Allowed tables"):
        agent.run(
            business_question="Find customers",
            database_schema=DATABASE_SCHEMA,
            allowed_tables=set(),
        )


def test_sql_agent_rejects_multiple_statements():
    response = valid_response()
    response["sql_query"] = (
        "SELECT * FROM customers; DROP TABLE customers"
    )

    agent = SQLAnalystAgent(
        client=FakeSQLLLMClient(response)
    )

    with pytest.raises(SQLValidationError):
        agent.run(
            business_question="Find all customers",
            database_schema=DATABASE_SCHEMA,
            allowed_tables={"customers"},
        )


def test_sql_agent_rejects_unapproved_table():
    response = valid_response()
    response["sql_query"] = "SELECT * FROM employee_salaries"
    response["tables_used"] = ["employee_salaries"]

    agent = SQLAnalystAgent(
        client=FakeSQLLLMClient(response)
    )

    with pytest.raises(
        SQLValidationError,
        match="not in the allowed schema",
    ):
        agent.run(
            business_question="Find all customers",
            database_schema=DATABASE_SCHEMA,
            allowed_tables={"customers"},
        )


def test_sql_agent_rejects_incorrect_reported_tables():
    response = valid_response()
    response["tables_used"] = ["orders"]

    agent = SQLAnalystAgent(
        client=FakeSQLLLMClient(response)
    )

    with pytest.raises(
        SQLValidationError,
        match="Reported tables do not match",
    ):
        agent.run(
            business_question="Find all customers",
            database_schema=DATABASE_SCHEMA,
            allowed_tables={"customers"},
        )


def test_sql_agent_rejects_mismatched_business_question():
    response = valid_response()
    response["business_question"] = "Find all orders"

    agent = SQLAnalystAgent(
        client=FakeSQLLLMClient(response)
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        agent.run(
            business_question="Find all customers",
            database_schema=DATABASE_SCHEMA,
            allowed_tables={"customers"},
        )


def test_sql_agent_rejects_invalid_structured_output():
    response = valid_response()
    response["is_read_only"] = False

    agent = SQLAnalystAgent(
        client=FakeSQLLLMClient(response)
    )

    with pytest.raises(ValueError):
        agent.run(
            business_question="Find all customers",
            database_schema=DATABASE_SCHEMA,
            allowed_tables={"customers"},
        )
