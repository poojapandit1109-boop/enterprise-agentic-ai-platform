
import pytest

from src.orchestration.graph import agent_graph
from src.orchestration.routing import supervisor_node

SAMPLE_RECORDS = [
    {"customer_id": 1, "monthly_spend": 100, "churned": 0},
    {"customer_id": 2, "monthly_spend": 200, "churned": 1},
    {"customer_id": 3, "monthly_spend": 300, "churned": 0},
    {"customer_id": 4, "monthly_spend": 400, "churned": 1},
]


@pytest.mark.parametrize(
    ("task", "expected_agent"),
    [
        ("Research market trends", "ResearchAgent"),
        ("Analyze quarterly revenue", "SQLAnalystAgent"),
        ("Calculate average customer churn", "PythonAnalystAgent"),
    ],
)
def test_supervisor_routes_to_expected_agent(task, expected_agent):
    result = supervisor_node({"user_task": task})

    assert result["next_agent"] == expected_agent
    assert result["workflow_status"] == "running"


@pytest.mark.parametrize(
    ("task", "expected_agent"),
    [
        ("Research market trends", "ResearchAgent"),
        ("Analyze quarterly revenue", "SQLAnalystAgent"),
        ("Calculate average customer churn", "PythonAnalystAgent"),
    ],
)
def test_agent_graph_executes_expected_agent(task, expected_agent):
    state = {
        "user_task": task,
        "agent_results": [],
        "errors": [],
    }

    if expected_agent == "PythonAnalystAgent":
        state["analysis_records"] = SAMPLE_RECORDS

    result = agent_graph.invoke(state)

    assert result["current_agent"] == expected_agent
    assert result["workflow_status"] == "completed"
    assert result["agent_results"][0]["agent_name"] == expected_agent
    assert result["agent_results"][0]["status"] == "completed"
    assert result["final_response"]


def test_python_analyst_requires_dataset():
    result = agent_graph.invoke(
        {
            "user_task": "Calculate average customer churn",
            "agent_results": [],
            "errors": [],
        }
    )

    assert result["current_agent"] == "PythonAnalystAgent"
    assert result["workflow_status"] == "failed"
    assert result["agent_results"] == []
    assert any(
        "requires analysis_records" in error
        for error in result["errors"]
    )


def test_python_analyst_returns_calculated_metrics():
    result = agent_graph.invoke(
        {
            "user_task": "Calculate average customer churn",
            "analysis_records": SAMPLE_RECORDS,
            "agent_results": [],
            "errors": [],
        }
    )

    output = result["agent_results"][0]["output"]

    assert result["workflow_status"] == "completed"
    assert output["key_metrics"]["mean__monthly_spend"] == 250.0
    assert output["key_metrics"]["mean__churned"] == 0.5


def test_supervisor_rejects_empty_task():
    result = supervisor_node({"user_task": ""})

    assert result["workflow_status"] == "failed"
    assert "User task cannot be empty." in result["errors"]


def test_agent_graph_routes_empty_task_to_end():
    result = agent_graph.invoke(
        {
            "user_task": "",
            "agent_results": [],
            "errors": [],
        }
    )

    assert result["workflow_status"] == "failed"
    assert "User task cannot be empty." in result["errors"]
    assert "final_response" not in result
