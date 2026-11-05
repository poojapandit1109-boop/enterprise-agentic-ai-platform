
import json

import pytest

from src.agents.report_mock_client import ReportMockLLMClient


def build_prompt(verification_status="passed"):
    context = {
        "user_task": "Analyze quarterly revenue",
        "verification_status": verification_status,
        "findings": [
            {
                "title": "Revenue analysis",
                "detail": "Revenue analysis was prepared.",
                "source_agent": "SQLAnalystAgent",
            }
        ],
        "limitations": [],
    }

    return "Report context:\n" + json.dumps(context)


def test_mock_client_generates_valid_executive_report():
    client = ReportMockLLMClient()

    response = client.generate(build_prompt())

    output = json.loads(response.content)

    assert response.is_mock is True
    assert response.model == "report-mock-llm"
    assert output["agent_name"] == "ReportGeneratorAgent"
    assert output["status"] == "completed"
    assert output["verification_status"] == "passed"
    assert len(output["key_findings"]) == 1
    assert output["recommendations"]


def test_mock_client_marks_review_report():
    client = ReportMockLLMClient()

    response = client.generate(
        build_prompt("needs_review")
    )

    output = json.loads(response.content)

    assert output["status"] == "needs_review"
    assert output["verification_status"] == "needs_review"
    assert "WARNING" in output["disclaimer"]


def test_mock_client_rejects_empty_prompt():
    client = ReportMockLLMClient()

    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        client.generate("")


def test_mock_client_rejects_missing_context():
    client = ReportMockLLMClient()

    with pytest.raises(ValueError, match="could not be extracted"):
        client.generate("Generate an executive report.")


def test_mock_client_rejects_invalid_json():
    client = ReportMockLLMClient()

    with pytest.raises(ValueError, match="valid JSON"):
        client.generate("Report context:\n{invalid json}")


def test_mock_client_rejects_unsupported_verification_status():
    client = ReportMockLLMClient()

    context = {
        "user_task": "Test",
        "verification_status": "rejected",
        "findings": [
            {
                "title": "Test",
                "detail": "Test finding",
                "source_agent": "SQLAnalystAgent",
            }
        ],
        "limitations": [],
    }

    prompt = "Report context:\n" + json.dumps(context)

    with pytest.raises(ValueError, match="Unsupported verification status"):
        client.generate(prompt)


def test_mock_client_rejects_empty_findings():
    client = ReportMockLLMClient()

    context = {
        "user_task": "Test",
        "verification_status": "passed",
        "findings": [],
        "limitations": [],
    }

    prompt = "Report context:\n" + json.dumps(context)

    with pytest.raises(ValueError, match="At least one finding"):
        client.generate(prompt)
