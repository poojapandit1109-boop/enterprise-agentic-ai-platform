
from langgraph.graph import END, START, StateGraph

from src.orchestration.routing import supervisor_node
from src.orchestration.state import AgentState


def research_agent_node(state: AgentState) -> dict:
    """Mock research agent."""

    return {
        "current_agent": "ResearchAgent",
        "agent_results": [
            {
                "agent_name": "ResearchAgent",
                "status": "completed",
                "output": {
                    "summary": (
                        f"Research completed for: {state['user_task']}"
                    )
                },
                "error": None,
            }
        ],
        "workflow_status": "completed",
        "final_response": (
            f"ResearchAgent completed the task: {state['user_task']}"
        ),
    }


def sql_analyst_node(state: AgentState) -> dict:
    """Mock SQL analyst agent."""

    return {
        "current_agent": "SQLAnalystAgent",
        "agent_results": [
            {
                "agent_name": "SQLAnalystAgent",
                "status": "completed",
                "output": {
                    "summary": (
                        f"SQL analysis completed for: {state['user_task']}"
                    )
                },
                "error": None,
            }
        ],
        "workflow_status": "completed",
        "final_response": (
            f"SQLAnalystAgent completed the task: {state['user_task']}"
        ),
    }


def python_analyst_node(state: AgentState) -> dict:
    """Mock Python analyst agent."""

    return {
        "current_agent": "PythonAnalystAgent",
        "agent_results": [
            {
                "agent_name": "PythonAnalystAgent",
                "status": "completed",
                "output": {
                    "summary": (
                        f"Python analysis completed for: {state['user_task']}"
                    )
                },
                "error": None,
            }
        ],
        "workflow_status": "completed",
        "final_response": (
            f"PythonAnalystAgent completed the task: {state['user_task']}"
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
