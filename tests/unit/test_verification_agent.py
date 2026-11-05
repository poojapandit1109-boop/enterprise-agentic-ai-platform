
import pytest

from src.agents.verification_agent import VerificationAgent


@pytest.fixture
def verifier():
    return VerificationAgent()


@pytest.fixture
def research_result():
    return {
        "agent_name": "ResearchAgent",
        "status": "completed",
        "output": {
            "agent_name": "ResearchAgent",
            "status": "completed",
            "topic": "Customer churn",
            "summary": "Customer churn increased during the quarter.",
            "findings": [
                {
                    "finding": "Churn increased.",
                    "evidence": "Synthetic sample analysis.",
                    "confidence_score": 0.92,
                }
            ],
            "missing_information": [],
        },
        "error": None,
    }


@pytest.fixture
def sql_result():
    return {
        "agent_name": "SQLAnalystAgent",
        "status": "completed",
        "output": {
            "agent_name": "SQLAnalystAgent",
            "status": "completed",
            "business_question": "How many customers are there?",
            "sql_query": "SELECT COUNT(*) FROM customers",
            "explanation": "Counts customer records.",
            "tables_used": ["customers"],
            "is_read_only": True,
        },
        "error": None,
    }


@pytest.fixture
def python_result():
    return {
        "agent_name": "PythonAnalystAgent",
        "status": "completed",
        "output": {
            "agent_name": "PythonAnalystAgent",
            "status": "completed",
            "objective": "Calculate average churn.",
            "analysis_summary": "Average churn was calculated.",
            "key_metrics": {
                "average_churn": 0.15,
                "customer_count": 100.0,
            },
            "insights": [
                "Average churn is 15 percent."
            ],
            "limitations": [],
        },
        "error": None,
    }


def test_valid_research_result_passes(
    verifier,
    research_result,
):
    result = verifier.run(research_result)

    assert result["agent_name"] == "VerificationAgent"
    assert result["status"] == "completed"
    assert result["output"]["status"] == "passed"
    assert result["output"]["verified_agent"] == "ResearchAgent"


def test_research_result_with_missing_information_needs_review(
    verifier,
    research_result,
):
    research_result["output"]["missing_information"] = [
        "Historical customer data is unavailable."
    ]

    result = verifier.run(research_result)

    assert result["status"] == "needs_review"
    assert result["output"]["status"] == "needs_review"
    assert len(result["output"]["warnings"]) > 0


def test_research_result_with_invalid_confidence_is_rejected(
    verifier,
    research_result,
):
    research_result["output"]["findings"][0]["confidence_score"] = 1.5

    result = verifier.run(research_result)

    assert result["status"] == "failed"
    assert result["output"]["status"] == "rejected"


def test_valid_sql_result_passes(
    verifier,
    sql_result,
):
    result = verifier.run(sql_result)

    assert result["status"] == "completed"
    assert result["output"]["status"] == "passed"
    assert result["output"]["verified_agent"] == "SQLAnalystAgent"


def test_sql_result_with_delete_query_is_rejected(
    verifier,
    sql_result,
):
    sql_result["output"]["sql_query"] = "DELETE FROM customers"

    result = verifier.run(sql_result)

    assert result["status"] == "failed"
    assert result["output"]["status"] == "rejected"


def test_sql_result_with_mismatched_tables_is_rejected(
    verifier,
    sql_result,
):
    sql_result["output"]["tables_used"] = ["orders"]

    result = verifier.run(sql_result)

    assert result["status"] == "failed"
    assert result["output"]["status"] == "rejected"


def test_sql_result_with_false_read_only_flag_is_rejected(
    verifier,
    sql_result,
):
    sql_result["output"]["is_read_only"] = False

    result = verifier.run(sql_result)

    assert result["status"] == "failed"
    assert result["output"]["status"] == "rejected"


def test_valid_python_result_passes(
    verifier,
    python_result,
):
    result = verifier.run(python_result)

    assert result["status"] == "completed"
    assert result["output"]["status"] == "passed"
    assert result["output"]["verified_agent"] == "PythonAnalystAgent"


def test_python_result_with_nan_metric_is_rejected(
    verifier,
    python_result,
):
    python_result["output"]["key_metrics"]["average_churn"] = float("nan")

    result = verifier.run(python_result)

    assert result["status"] == "failed"
    assert result["output"]["status"] == "rejected"


def test_python_result_with_invalid_insights_is_rejected(
    verifier,
    python_result,
):
    python_result["output"]["insights"] = []

    result = verifier.run(python_result)

    assert result["status"] == "failed"
    assert result["output"]["status"] == "rejected"


def test_specialist_agent_needing_review_needs_review(
    verifier,
    python_result,
):
    python_result["status"] = "needs_review"

    result = verifier.run(python_result)

    assert result["status"] == "needs_review"
    assert result["output"]["status"] == "needs_review"


def test_failed_specialist_agent_is_rejected(
    verifier,
    python_result,
):
    python_result["status"] = "failed"

    result = verifier.run(python_result)

    assert result["status"] == "failed"
    assert result["output"]["status"] == "rejected"


def test_mismatched_agent_identity_is_rejected(
    verifier,
    research_result,
):
    research_result["output"]["agent_name"] = "SQLAnalystAgent"

    result = verifier.run(research_result)

    assert result["status"] == "failed"
    assert result["output"]["status"] == "rejected"


def test_unsupported_agent_is_rejected_with_exception(verifier):
    result = {
        "agent_name": "UnknownAgent",
        "status": "completed",
        "output": {},
        "error": None,
    }

    with pytest.raises(ValueError, match="Unsupported agent name"):
        verifier.run(result)


def test_non_dictionary_input_is_rejected(verifier):
    with pytest.raises(TypeError, match="must be a dictionary"):
        verifier.run("invalid")
