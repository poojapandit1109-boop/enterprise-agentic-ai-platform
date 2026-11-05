
from langgraph.graph import END, START, StateGraph

from src.agents.python_analyst_agent import PythonAnalystAgent
from src.agents.report_generator_agent import ReportGeneratorAgent
from src.agents.research_agent import ResearchAgent
from src.agents.sql_analyst_agent import SQLAnalystAgent
from src.agents.verification_agent import VerificationAgent
from src.orchestration.routing import supervisor_node
from src.orchestration.state import AgentState

DEFAULT_DATABASE_SCHEMA = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100),
    city VARCHAR(100)
);
"""

DEFAULT_ALLOWED_TABLES = ["customers"]


def research_agent_node(state: AgentState) -> dict:
    """Execute the functional ResearchAgent."""

    agent = ResearchAgent()
    result = agent.run(state["user_task"])

    summary = result["output"]["summary"]

    return {
        "current_agent": "ResearchAgent",
        "agent_results": [result],
        "workflow_status": (
            "completed"
            if result["status"] == "completed"
            else "needs_review"
        ),
        "final_response": summary,
    }


def sql_analyst_node(state: AgentState) -> dict:
    """Execute the functional SQL Analyst Agent."""

    agent = SQLAnalystAgent()

    database_schema = state.get(
        "database_schema",
        DEFAULT_DATABASE_SCHEMA,
    )

    allowed_tables = state.get(
        "allowed_tables",
        DEFAULT_ALLOWED_TABLES,
    )

    result = agent.run(
        business_question=state["user_task"],
        database_schema=database_schema,
        allowed_tables=set(allowed_tables),
    )

    output = result["output"]

    final_response = (
        f"SQL analysis prepared for: {output['business_question']}\n"
        f"SQL query: {output['sql_query']}\n"
        f"Explanation: {output['explanation']}\n"
        "Note: The query was validated but not executed."
    )

    return {
        "current_agent": "SQLAnalystAgent",
        "agent_results": [result],
        "workflow_status": (
            "completed"
            if result["status"] == "completed"
            else "needs_review"
        ),
        "final_response": final_response,
    }


def python_analyst_node(state: AgentState) -> dict:
    """Execute the functional Pandas-based Python Analyst Agent."""

    records = state.get("analysis_records")

    if not records:
        return {
            "current_agent": "PythonAnalystAgent",
            "agent_results": [],
            "workflow_status": "failed",
            "errors": state.get("errors", [])
            + [
                (
                    "Python Analyst requires analysis_records. "
                    "Supply a non-empty dataset."
                )
            ],
        }

    objective = state.get(
        "analysis_objective",
        state["user_task"],
    )

    agent = PythonAnalystAgent()

    result = agent.run(
        objective=objective,
        records=records,
    )

    output = result["output"]

    final_response = (
        "Python analysis completed.\n"
        f"Objective: {output['objective']}\n"
        f"Summary: {output['analysis_summary']}\n"
        f"Metrics: {output['key_metrics']}\n"
        f"Insights: {'; '.join(output['insights'])}"
    )

    return {
        "current_agent": "PythonAnalystAgent",
        "agent_results": [result],
        "workflow_status": (
            "completed"
            if result["status"] == "completed"
            else "needs_review"
        ),
        "final_response": final_response,
    }


def verification_node(state: AgentState) -> dict:
    """Independently verify the most recent specialist-agent result."""

    agent_results = state.get("agent_results", [])

    if not agent_results:
        return {
            "workflow_status": "failed",
            "errors": state.get("errors", [])
            + ["Verification skipped because no specialist result exists."],
        }

    specialist_result = agent_results[-1]

    verifier = VerificationAgent()
    verification_result = verifier.run(specialist_result)

    verification_output = verification_result["output"]
    verification_status = verification_output["status"]

    workflow_status = {
        "passed": "completed",
        "rejected": "failed",
        "needs_review": "needs_review",
    }[verification_status]

    existing_response = state.get("final_response", "")

    verification_summary = (
        "\n\nVerification: "
        f"{verification_output['summary']}"
    )

    warnings = verification_output.get("warnings", [])

    if warnings:
        warning_text = "\n".join(
            f"- {warning}" for warning in warnings
        )
        verification_summary += (
            "\nVerification warnings:\n"
            f"{warning_text}"
        )

    errors = state.get("errors", [])

    if verification_status == "rejected":
        errors = errors + [
            verification_output["summary"]
        ]

    return {
        "verification_result": verification_result,
        "workflow_status": workflow_status,
        "errors": errors,
        "final_response": existing_response + verification_summary,
    }


def route_after_verification(state: AgentState) -> str:
    """Allow only passed or reviewable outputs into report generation."""

    verification_result = state.get("verification_result")

    if not verification_result:
        return "end"

    verification_output = verification_result.get("output", {})

    if not isinstance(verification_output, dict):
        return "end"

    verification_status = verification_output.get("status")

    if verification_status not in {"passed", "needs_review"}:
        return "end"

    if verification_result.get("status") not in {
        "completed",
        "needs_review",
    }:
        return "end"

    if state.get("workflow_status") == "failed":
        return "end"

    return "report_generator"


def report_generator_node(state: AgentState) -> dict:
    """Generate an executive report from verified specialist output."""

    agent_results = state.get("agent_results", [])
    verification_result = state.get("verification_result")

    if not agent_results or not verification_result:
        return {
            "workflow_status": "failed",
            "errors": state.get("errors", [])
            + [
                (
                    "Report generation requires specialist output "
                    "and verification results."
                )
            ],
        }

    specialist_result = agent_results[-1]

    agent = ReportGeneratorAgent()

    report_result = agent.run(
        user_task=state["user_task"],
        specialist_result=specialist_result,
        verification_result=verification_result,
    )

    report = report_result["output"]

    report_text = (
        "\n\n"
        f"EXECUTIVE REPORT: {report['report_title']}\n\n"
        f"Executive Summary:\n{report['executive_summary']}\n\n"
        "Key Findings:\n"
    )

    for finding in report["key_findings"]:
        report_text += (
            f"- {finding['title']}: {finding['detail']} "
            f"(Source: {finding['source_agent']})\n"
        )

    report_text += "\nRecommendations:\n"

    for recommendation in report["recommendations"]:
        report_text += f"- {recommendation}\n"

    if report["limitations"]:
        report_text += "\nLimitations:\n"

        for limitation in report["limitations"]:
            report_text += f"- {limitation}\n"

    report_text += (
        f"\nVerification Status: {report['verification_status']}\n"
        f"Disclaimer: {report['disclaimer']}"
    )

    return {
        "current_agent": "ReportGeneratorAgent",
        "report_result": report_result,
        "workflow_status": (
            "completed"
            if report_result["status"] == "completed"
            else "needs_review"
        ),
        "final_response": (
            state.get("final_response", "") + report_text
        ),
    }


def route_to_agent(state: AgentState) -> str:
    """Select the next graph node based on supervisor routing."""

    next_agent = state.get("next_agent")

    routes = {
        "ResearchAgent": "research",
        "SQLAnalystAgent": "sql_analyst",
        "PythonAnalystAgent": "python_analyst",
    }

    return routes.get(next_agent, "end")


def build_agent_graph():
    """Build and compile the multi-agent reporting workflow."""

    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research", research_agent_node)
    workflow.add_node("sql_analyst", sql_analyst_node)
    workflow.add_node("python_analyst", python_analyst_node)
    workflow.add_node("verification", verification_node)
    workflow.add_node("report_generator", report_generator_node)

    workflow.add_edge(START, "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "research": "research",
            "sql_analyst": "sql_analyst",
            "python_analyst": "python_analyst",
            "end": END,
        },
    )

    workflow.add_edge("research", "verification")
    workflow.add_edge("sql_analyst", "verification")
    workflow.add_edge("python_analyst", "verification")

    workflow.add_conditional_edges(
        "verification",
        route_after_verification,
        {
            "report_generator": "report_generator",
            "end": END,
        },
    )

    workflow.add_edge("report_generator", END)

    return workflow.compile()


agent_graph = build_agent_graph()
