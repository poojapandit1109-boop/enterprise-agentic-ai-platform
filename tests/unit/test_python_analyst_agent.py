
import json

import pytest

from src.agents.python_analyst_agent import PythonAnalystAgent
from src.llm.base import BaseLLMClient, LLMResponse

SAMPLE_RECORDS = [
    {"customer_id": 1, "monthly_spend": 100, "churned": 0},
    {"customer_id": 2, "monthly_spend": 200, "churned": 1},
    {"customer_id": 3, "monthly_spend": 300, "churned": 0},
    {"customer_id": 4, "monthly_spend": 400, "churned": 1},
]


class InvalidPythonMockClient(BaseLLMClient):
    """Test client that returns deliberately invalid JSON."""

    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(
            content="not valid JSON",
            model="invalid-test-client",
            is_mock=True,
        )


class WrongMetricsMockClient(BaseLLMClient):
    """Test client that returns metrics different from Pandas results."""

    def generate(self, prompt: str) -> LLMResponse:
        start = prompt.index("Analysis objective:") + len(
            "Analysis objective:"
        )
        end = prompt.index("\n\nAvailable data description:")
        objective = prompt[start:end].strip()

        response = {
            "agent_name": "PythonAnalystAgent",
            "status": "completed",
            "objective": objective,
            "analysis_summary": "Test analysis.",
            "key_metrics": {
                "mean__monthly_spend": 999.0
            },
            "insights": ["Test insight."],
            "limitations": [],
        }

        return LLMResponse(
            content=json.dumps(response),
            model="wrong-metrics-test-client",
            is_mock=True,
        )


def test_python_analyst_returns_structured_result():
    agent = PythonAnalystAgent()

    result = agent.run(
        objective="Analyze customer spending and churn",
        records=SAMPLE_RECORDS,
    )

    assert result["agent_name"] == "PythonAnalystAgent"
    assert result["status"] == "completed"
    assert result["error"] is None
    assert result["output"]["objective"] == (
        "Analyze customer spending and churn"
    )


def test_python_analyst_returns_actual_pandas_metrics():
    agent = PythonAnalystAgent()

    result = agent.run(
        objective="Calculate customer spending statistics",
        records=SAMPLE_RECORDS,
    )

    metrics = result["output"]["key_metrics"]

    assert metrics["mean__monthly_spend"] == 250.0
    assert metrics["median__monthly_spend"] == 250.0
    assert metrics["mean__churned"] == 0.5


def test_python_analyst_includes_analysis_limitations():
    agent = PythonAnalystAgent()

    result = agent.run(
        objective="Analyze customer churn",
        records=SAMPLE_RECORDS,
    )

    limitations = result["output"]["limitations"]

    assert any(
        "Correlation does not establish causation" in item
        for item in limitations
    )


def test_python_analyst_rejects_empty_objective():
    agent = PythonAnalystAgent()

    with pytest.raises(ValueError, match="objective"):
        agent.run(
            objective=" ",
            records=SAMPLE_RECORDS,
        )


def test_python_analyst_rejects_empty_records():
    agent = PythonAnalystAgent()

    with pytest.raises(
        ValueError,
        match="dataset cannot be empty",
    ):
        agent.run(
            objective="Analyze customer churn",
            records=[],
        )


def test_python_analyst_rejects_invalid_json_response():
    agent = PythonAnalystAgent(
        client=InvalidPythonMockClient()
    )

    with pytest.raises(ValueError):
        agent.run(
            objective="Analyze customer churn",
            records=SAMPLE_RECORDS,
        )


def test_python_analyst_rejects_metrics_not_matching_pandas():
    agent = PythonAnalystAgent(
        client=WrongMetricsMockClient()
    )

    with pytest.raises(ValueError, match="metrics"):
        agent.run(
            objective="Analyze customer spending",
            records=SAMPLE_RECORDS,
        )
