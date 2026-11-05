import pytest

from src.agents.report_generator_agent import ReportGeneratorAgent


def build_specialist_result(agent_name, output, status="completed"):
    return {
        "agent_name": agent_name,
        "status": status,
        "output": {
            "agent_name": agent_name,
            **output,
        },
        "error": None,
    }


def build_verification_result(
    agent_name,
    verification_status="passed",
):
    agent_status = {
        "passed": "completed",
        "rejected": "failed",
        "needs_review": "needs_review",
    }[verification_status]

    return {
        "agent_name": "VerificationAgent",
        "status": agent_status,
        "output": {
            "agent_name": "VerificationAgent",
            "status": verification_status,
            "verified_agent": agent_name,
            "checks": [],
            "summary": f"Verification {verification_status}.",
            "warnings": (
                ["Independent evidence review is required."]
                if verification_status == "needs_review"
                else []
            ),
        },
        "error": None,
    }


def test_report_generator_creates_report_from_research():
    agent = ReportGeneratorAgent()

    specialist = build_specialist_result(
        "ResearchAgent",
        {
            "topic": "Market trends",
            "summary": "Simulated market summary.",
            "findings": [
                {
                    "finding": "A simulated market finding.",
                    "evidence": "Synthetic demonstration evidence.",
                    "confidence_score": 0.85,
                }
            ],
            "missing_information": [],
        },
    )

    verification = build_verification_result("ResearchAgent")

    result = agent.run(
        "Research market trends",
        specialist,
        verification,
    )

    assert result["agent_name"] == "ReportGeneratorAgent"
    assert result["status"] == "completed"
    assert result["output"]["verification_status"] == "passed"
    assert result["output"]["key_findings"][0]["source_agent"] == (
        "ResearchAgent"
    )


def test_report_generator_creates_report_from_sql():
    agent = ReportGeneratorAgent()

    specialist = build_specialist_result(
        "SQLAnalystAgent",
        {
            "business_question": "Analyze revenue",
            "sql_query": "SELECT * FROM customers",
            "explanation": "Retrieves customer records.",
            "tables_used": ["customers"],
            "is_read_only": True,
        },
    )

    verification = build_verification_result("SQLAnalystAgent")

    result = agent.run(
        "Analyze quarterly revenue",
        specialist,
        verification,
    )

    report = result["output"]

    assert result["status"] == "completed"
    assert report["verification_status"] == "passed"
    assert any(
        "not executed" in limitation
        for limitation in report["limitations"]
    )


def test_report_generator_creates_report_from_python():
    agent = ReportGeneratorAgent()

    specialist = build_specialist_result(
        "PythonAnalystAgent",
        {
            "objective": "Analyze customer churn",
            "analysis_summary": "Customer churn analysis.",
            "key_metrics": {
                "mean_monthly_spend": 250.0,
                "churn_rate": 0.5,
            },
            "insights": ["Half of the sample customers churned."],
            "limitations": [],
        },
    )

    verification = build_verification_result("PythonAnalystAgent")

    result = agent.run(
        "Analyze customer churn",
        specialist,
        verification,
    )

    report = result["output"]

    assert result["status"] == "completed"
    assert report["verification_status"] == "passed"
    assert len(report["key_findings"]) == 3


def test_report_generator_preserves_needs_review_status():
    agent = ReportGeneratorAgent()

    specialist = build_specialist_result(
        "ResearchAgent",
        {
            "topic": "Market trends",
            "summary": "Simulated market summary.",
            "findings": [
                {
                    "finding": "A simulated finding.",
                    "evidence": "Synthetic evidence.",
                    "confidence_score": 0.85,
                }
            ],
            "missing_information": [
                "Independent source verification is pending."
            ],
        },
    )

    verification = build_verification_result(
        "ResearchAgent",
        "needs_review",
    )

    result = agent.run(
        "Research market trends",
        specialist,
        verification,
    )

    report = result["output"]

    assert result["status"] == "needs_review"
    assert report["verification_status"] == "needs_review"
    assert "WARNING" in report["disclaimer"]
    assert any(
        "Independent evidence review is required" in limitation
        for limitation in report["limitations"]
    )


def test_report_generator_blocks_rejected_output():
    agent = ReportGeneratorAgent()

    specialist = build_specialist_result(
        "SQLAnalystAgent",
        {
            "business_question": "Analyze revenue",
            "sql_query": "SELECT * FROM customers",
            "explanation": "Example query.",
            "tables_used": ["customers"],
            "is_read_only": True,
        },
    )

    verification = build_verification_result(
        "SQLAnalystAgent",
        "rejected",
    )

    with pytest.raises(ValueError, match="output was rejected"):
        agent.run("Analyze revenue", specialist, verification)


def test_report_generator_rejects_mismatched_verification_agent():
    agent = ReportGeneratorAgent()

    specialist = build_specialist_result(
        "SQLAnalystAgent",
        {
            "business_question": "Analyze revenue",
            "sql_query": "SELECT * FROM customers",
            "explanation": "Example query.",
            "tables_used": ["customers"],
            "is_read_only": True,
        },
    )

    verification = build_verification_result("ResearchAgent")

    with pytest.raises(ValueError, match="does not match"):
        agent.run("Analyze revenue", specialist, verification)


def test_report_generator_rejects_failed_specialist():
    agent = ReportGeneratorAgent()

    specialist = build_specialist_result(
        "SQLAnalystAgent",
        {
            "business_question": "Analyze revenue",
            "sql_query": "SELECT * FROM customers",
            "explanation": "Example query.",
            "tables_used": ["customers"],
            "is_read_only": True,
        },
        status="failed",
    )

    verification = build_verification_result("SQLAnalystAgent")

    with pytest.raises(ValueError, match="completed specialist"):
        agent.run("Analyze revenue", specialist, verification)


def test_report_generator_rejects_empty_task():
    agent = ReportGeneratorAgent()

    with pytest.raises(ValueError, match="User task cannot be empty"):
        agent.run("", {}, {})
