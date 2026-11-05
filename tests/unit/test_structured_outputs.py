
import json

import pytest
from pydantic import ValidationError

from src.llm.base import BaseLLMClient, LLMResponse
from src.llm.output_parser import (
    StructuredOutputError,
    parse_structured_output,
)
from src.llm.structured_client import generate_structured_output
from src.llm.structured_outputs import (
    PythonAnalysisOutput,
    ResearchOutput,
    SQLAnalysisOutput,
)


# --------------------------------------------------
# Research Agent tests
# --------------------------------------------------

def test_valid_research_output():
    data = {
        "status": "completed",
        "topic": "Revenue decline",
        "summary": "Revenue declined by 10 percent.",
        "findings": [
            {
                "finding": "Revenue decreased.",
                "evidence": "Quarterly report",
                "confidence_score": 0.95,
            }
        ],
    }

    result = ResearchOutput.model_validate(data)

    assert result.agent_name == "ResearchAgent"
    assert result.status == "completed"
    assert result.findings[0].confidence_score == 0.95


def test_research_confidence_must_be_between_zero_and_one():
    data = {
        "status": "completed",
        "topic": "Revenue decline",
        "summary": "Revenue declined.",
        "findings": [
            {
                "finding": "Revenue decreased.",
                "evidence": "Quarterly report",
                "confidence_score": 1.5,
            }
        ],
    }

    with pytest.raises(ValidationError):
        ResearchOutput.model_validate(data)


# --------------------------------------------------
# SQL Analyst tests
# --------------------------------------------------

def test_valid_sql_output():
    data = {
        "status": "completed",
        "business_question": "Find quarterly revenue",
        "sql_query": "SELECT SUM(revenue) FROM sales",
        "explanation": "Calculates total revenue.",
        "tables_used": ["sales"],
        "is_read_only": True,
    }

    result = SQLAnalysisOutput.model_validate(data)

    assert result.agent_name == "SQLAnalystAgent"
    assert result.is_read_only is True


def test_sql_output_rejects_delete_query():
    data = {
        "status": "completed",
        "business_question": "Delete old records",
        "sql_query": "DELETE FROM sales",
        "explanation": "Delete records.",
        "tables_used": ["sales"],
        "is_read_only": True,
    }

    with pytest.raises(ValidationError):
        SQLAnalysisOutput.model_validate(data)


def test_sql_output_rejects_false_read_only_flag():
    data = {
        "status": "completed",
        "business_question": "Find revenue",
        "sql_query": "SELECT revenue FROM sales",
        "explanation": "Reads revenue.",
        "tables_used": ["sales"],
        "is_read_only": False,
    }

    with pytest.raises(ValidationError):
        SQLAnalysisOutput.model_validate(data)


# --------------------------------------------------
# Python Analyst tests
# --------------------------------------------------

def test_valid_python_analysis_output():
    data = {
        "status": "completed",
        "objective": "Analyze revenue",
        "analysis_summary": "Revenue declined.",
        "key_metrics": {
            "revenue_change_percent": -10.0
        },
        "insights": [
            "Revenue declined by 10 percent."
        ],
    }

    result = PythonAnalysisOutput.model_validate(data)

    assert result.agent_name == "PythonAnalystAgent"
    assert result.key_metrics["revenue_change_percent"] == -10.0


def test_python_output_rejects_invalid_metric_type():
    data = {
        "status": "completed",
        "objective": "Analyze revenue",
        "analysis_summary": "Revenue declined.",
        "key_metrics": {
            "revenue_change_percent": "unknown"
        },
        "insights": ["Revenue declined."],
    }

    with pytest.raises(ValidationError):
        PythonAnalysisOutput.model_validate(data)


# --------------------------------------------------
# Schema validation tests
# --------------------------------------------------

def test_unknown_fields_are_rejected():
    data = {
        "status": "completed",
        "topic": "Revenue decline",
        "summary": "Revenue declined.",
        "findings": [
            {
                "finding": "Revenue decreased.",
                "evidence": "Quarterly report",
                "confidence_score": 0.9,
            }
        ],
        "unexpected_field": "not allowed",
    }

    with pytest.raises(ValidationError):
        ResearchOutput.model_validate(data)


# --------------------------------------------------
# JSON parser tests
# --------------------------------------------------

def test_json_parser_returns_validated_object():
    data = {
        "status": "completed",
        "topic": "Revenue decline",
        "summary": "Revenue declined.",
        "findings": [
            {
                "finding": "Revenue decreased.",
                "evidence": "Quarterly report",
                "confidence_score": 0.9,
            }
        ],
    }

    result = parse_structured_output(
        json.dumps(data),
        ResearchOutput,
    )

    assert isinstance(result, ResearchOutput)
    assert result.topic == "Revenue decline"


def test_json_parser_rejects_invalid_json():
    with pytest.raises(StructuredOutputError, match="Invalid JSON"):
        parse_structured_output(
            "{invalid json}",
            ResearchOutput,
        )


def test_json_parser_rejects_empty_response():
    with pytest.raises(StructuredOutputError, match="cannot be empty"):
        parse_structured_output(
            "",
            ResearchOutput,
        )


def test_json_parser_rejects_json_array():
    with pytest.raises(StructuredOutputError, match="JSON object"):
        parse_structured_output(
            "[]",
            ResearchOutput,
        )


def test_json_parser_rejects_invalid_schema():
    data = {
        "status": "completed",
        "topic": "Revenue decline",
    }

    with pytest.raises(
        StructuredOutputError,
        match="validation failed",
    ):
        parse_structured_output(
            json.dumps(data),
            ResearchOutput,
        )


# --------------------------------------------------
# Mock LLM integration tests
# --------------------------------------------------

class FakeStructuredLLMClient(BaseLLMClient):
    """Test client returning valid JSON without API calls."""

    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(
            content=json.dumps({
                "status": "completed",
                "topic": "Revenue decline",
                "summary": "Revenue declined by 10 percent.",
                "findings": [
                    {
                        "finding": "Revenue decreased.",
                        "evidence": "Quarterly report",
                        "confidence_score": 0.95,
                    }
                ],
            }),
            model="mock-structured-llm",
            is_mock=True,
        )


def test_structured_client_returns_validated_output():
    client = FakeStructuredLLMClient()

    result = generate_structured_output(
        client=client,
        prompt="Analyze revenue decline",
        schema=ResearchOutput,
    )

    assert isinstance(result, ResearchOutput)
    assert result.topic == "Revenue decline"
    assert result.findings[0].confidence_score == 0.95


def test_structured_client_uses_mock_response():
    client = FakeStructuredLLMClient()

    response = client.generate("Test prompt")

    assert response.is_mock is True
    assert response.model == "mock-structured-llm"
