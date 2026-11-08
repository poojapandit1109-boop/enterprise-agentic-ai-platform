
"""Centralized least-privilege tool access policies for platform agents."""

from types import MappingProxyType

# Only explicitly listed tools can be accessed by each agent.
# These are policy identifiers; actual tool implementations will be
# connected to this policy in a later step.

AGENT_TOOL_PERMISSIONS = MappingProxyType(
    {
        "ResearchAgent": frozenset(
            {
                "web_search",
                "document_retrieval",
            }
        ),
        "SQLAnalystAgent": frozenset(
            {
                "read_only_sql",
            }
        ),
        "PythonAnalystAgent": frozenset(
            {
                "dataframe_analysis",
            }
        ),
        "VerificationAgent": frozenset(),
        "ReportGeneratorAgent": frozenset(),
    }
)


def is_tool_allowed(agent_name: str, tool_name: str) -> bool:
    """Return whether an agent is explicitly permitted to use a tool."""

    if not isinstance(agent_name, str) or not agent_name.strip():
        return False

    if not isinstance(tool_name, str) or not tool_name.strip():
        return False

    allowed_tools = AGENT_TOOL_PERMISSIONS.get(agent_name)

    if allowed_tools is None:
        return False

    return tool_name in allowed_tools


def require_tool_permission(agent_name: str, tool_name: str) -> None:
    """Raise PermissionError when an agent lacks permission for a tool."""

    if not is_tool_allowed(agent_name, tool_name):
        raise PermissionError(
            f"Agent '{agent_name}' is not permitted to use "
            f"tool '{tool_name}'."
        )


def get_allowed_tools(agent_name: str) -> frozenset[str]:
    """Return the immutable set of tools allowed for an agent."""

    if not isinstance(agent_name, str) or not agent_name.strip():
        return frozenset()

    return AGENT_TOOL_PERMISSIONS.get(agent_name, frozenset())
