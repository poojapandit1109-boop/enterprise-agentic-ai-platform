
import pytest

from src.security.tool_permissions import (
    AGENT_TOOL_PERMISSIONS,
    get_allowed_tools,
    is_tool_allowed,
    require_tool_permission,
)


@pytest.mark.parametrize(
    ("agent_name", "tool_name"),
    [
        ("ResearchAgent", "web_search"),
        ("ResearchAgent", "document_retrieval"),
        ("SQLAnalystAgent", "read_only_sql"),
        ("PythonAnalystAgent", "dataframe_analysis"),
    ],
)
def test_authorized_agent_can_access_allowed_tool(
    agent_name: str,
    tool_name: str,
) -> None:
    assert is_tool_allowed(agent_name, tool_name) is True


@pytest.mark.parametrize(
    ("agent_name", "tool_name"),
    [
        ("SQLAnalystAgent", "web_search"),
        ("SQLAnalystAgent", "document_retrieval"),
        ("ResearchAgent", "read_only_sql"),
        ("PythonAnalystAgent", "read_only_sql"),
        ("UnknownAgent", "read_only_sql"),
        ("ReportGeneratorAgent", "read_only_sql"),
        ("VerificationAgent", "dataframe_analysis"),
    ],
)
def test_unauthorized_agent_is_denied(
    agent_name: str,
    tool_name: str,
) -> None:
    assert is_tool_allowed(agent_name, tool_name) is False


@pytest.mark.parametrize(
    ("agent_name", "tool_name"),
    [
        ("", "web_search"),
        ("ResearchAgent", ""),
        ("   ", "web_search"),
        ("ResearchAgent", "   "),
        (None, "web_search"),
        ("ResearchAgent", None),
        (123, "web_search"),
        ("ResearchAgent", 123),
    ],
)
def test_invalid_inputs_are_denied(
    agent_name: str,
    tool_name: str,
) -> None:
    assert is_tool_allowed(agent_name, tool_name) is False


def test_require_tool_permission_allows_authorized_tool() -> None:
    require_tool_permission("SQLAnalystAgent", "read_only_sql")


def test_require_tool_permission_rejects_unauthorized_tool() -> None:
    with pytest.raises(PermissionError):
        require_tool_permission("SQLAnalystAgent", "web_search")


def test_require_tool_permission_rejects_unknown_agent() -> None:
    with pytest.raises(PermissionError):
        require_tool_permission("UnknownAgent", "read_only_sql")


def test_get_allowed_tools_returns_expected_permissions() -> None:
    assert get_allowed_tools("ResearchAgent") == frozenset(
        {"web_search", "document_retrieval"}
    )

    assert get_allowed_tools("SQLAnalystAgent") == frozenset(
        {"read_only_sql"}
    )

    assert get_allowed_tools("PythonAnalystAgent") == frozenset(
        {"dataframe_analysis"}
    )


def test_unknown_agent_receives_no_permissions() -> None:
    assert get_allowed_tools("UnknownAgent") == frozenset()


def test_report_generator_has_no_direct_tool_permissions() -> None:
    assert get_allowed_tools("ReportGeneratorAgent") == frozenset()


def test_permission_registry_is_immutable() -> None:
    with pytest.raises(TypeError):
        AGENT_TOOL_PERMISSIONS["UnknownAgent"] = frozenset({"web_search"})
