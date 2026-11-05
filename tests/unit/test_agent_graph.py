
import pytest

from src.orchestration.graph import agent_graph
from src.orchestration.routing import supervisor_node


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
    result = agent_graph.invoke(
        {
            "user_task": task,
            "agent_results": [],
            "errors": [],
        }
    )

    assert result["current_agent"] == expected_agent
    assert result["workflow_status"] == "completed"
    assert result["agent_results"][0]["agent_name"] == expected_agent
    assert result["agent_results"][0]["status"] == "completed"
    assert result["final_response"]


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
