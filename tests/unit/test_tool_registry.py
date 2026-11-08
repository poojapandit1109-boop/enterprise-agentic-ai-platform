
from unittest.mock import Mock

import pytest

from src.tools.tool_registry import ToolRegistry


def test_register_tool_successfully() -> None:
    registry = ToolRegistry()

    registry.register("read_only_sql", lambda query: query)

    assert registry.is_registered("read_only_sql") is True


def test_authorized_agent_can_execute_registered_tool() -> None:
    registry = ToolRegistry()
    mock_tool = Mock(return_value={"result": "success"})

    registry.register("read_only_sql", mock_tool)

    result = registry.execute(
        "SQLAnalystAgent",
        "read_only_sql",
        query="SELECT 1",
    )

    assert result == {"result": "success"}
    mock_tool.assert_called_once_with(query="SELECT 1")


def test_unauthorized_agent_cannot_execute_tool() -> None:
    registry = ToolRegistry()
    mock_tool = Mock()

    registry.register("read_only_sql", mock_tool)

    with pytest.raises(PermissionError):
        registry.execute(
            "ResearchAgent",
            "read_only_sql",
            query="SELECT 1",
        )

    mock_tool.assert_not_called()


def test_unknown_agent_cannot_execute_tool() -> None:
    registry = ToolRegistry()
    mock_tool = Mock()

    registry.register("read_only_sql", mock_tool)

    with pytest.raises(PermissionError):
        registry.execute(
            "UnknownAgent",
            "read_only_sql",
            query="SELECT 1",
        )

    mock_tool.assert_not_called()


def test_unregistered_tool_is_rejected() -> None:
    registry = ToolRegistry()

    with pytest.raises(LookupError):
        registry.execute(
            "SQLAnalystAgent",
            "read_only_sql",
            query="SELECT 1",
        )


def test_duplicate_tool_registration_is_rejected() -> None:
    registry = ToolRegistry()

    registry.register("read_only_sql", lambda: "first")

    with pytest.raises(ValueError, match="already registered"):
        registry.register("read_only_sql", lambda: "second")


@pytest.mark.parametrize(
    "tool_name",
    [
        "",
        "   ",
    ],
)
def test_empty_tool_name_is_rejected(tool_name: str) -> None:
    registry = ToolRegistry()

    with pytest.raises(ValueError):
        registry.register(tool_name, lambda: None)


def test_non_callable_tool_is_rejected() -> None:
    registry = ToolRegistry()

    with pytest.raises(TypeError):
        registry.register("read_only_sql", "not_callable")


def test_registered_tools_view_is_read_only() -> None:
    registry = ToolRegistry()
    registry.register("read_only_sql", lambda: None)

    with pytest.raises(TypeError):
        registry.registered_tools["web_search"] = lambda: None


def test_registered_tools_contains_only_registered_names() -> None:
    registry = ToolRegistry()

    registry.register("read_only_sql", lambda: None)
    registry.register("web_search", lambda: None)

    assert set(registry.registered_tools) == {
        "read_only_sql",
        "web_search",
    }


def test_research_agent_can_execute_authorized_search_tool() -> None:
    registry = ToolRegistry()
    mock_tool = Mock(return_value={"results": []})

    registry.register("web_search", mock_tool)

    result = registry.execute(
        "ResearchAgent",
        "web_search",
        query="Enterprise AI security",
    )

    assert result == {"results": []}
    mock_tool.assert_called_once_with(
        query="Enterprise AI security"
    )
