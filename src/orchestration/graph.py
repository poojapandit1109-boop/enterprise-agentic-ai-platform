
from langgraph.graph import END, START, StateGraph

from src.agents.python_analyst_agent import PythonAnalystAgent
from src.agents.research_agent import ResearchAgent
from src.agents.sql_analyst_agent import SQLAnalystAgent
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
            "errors": [
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
        f"Python analysis completed.\n"
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
    """Build and compile the multi-agent LangGraph workflow."""

    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research", research_agent_node)
    workflow.add_node("sql_analyst", sql_analyst_node)
    workflow.add_node("python_analyst", python_analyst_node)

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

    workflow.add_edge("research", END)
    workflow.add_edge("sql_analyst", END)
    workflow.add_edge("python_analyst", END)

    return workflow.compile()


agent_graph = build_agent_graph()
