
"""Secure execution gateway for registered agent tools."""

from collections.abc import Callable
from types import MappingProxyType
from typing import Any

from src.security.tool_permissions import require_tool_permission

ToolFunction = Callable[..., Any]


class ToolRegistry:
    """Register tools and enforce agent permissions before execution."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolFunction] = {}

    def register(
        self,
        tool_name: str,
        tool_function: ToolFunction,
    ) -> None:
        """Register a trusted tool implementation."""

        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("Tool name must be a non-empty string.")

        if not callable(tool_function):
            raise TypeError("Tool implementation must be callable.")

        if tool_name in self._tools:
            raise ValueError(f"Tool '{tool_name}' is already registered.")

        self._tools[tool_name] = tool_function

    def execute(
        self,
        agent_name: str,
        tool_name: str,
        **kwargs: Any,
    ) -> Any:
        """Authorize and execute a registered tool."""

        # Check authorization before looking up or executing the tool.
        require_tool_permission(agent_name, tool_name)

        tool_function = self._tools.get(tool_name)

        if tool_function is None:
            raise LookupError(
                f"Tool '{tool_name}' is not registered."
            )

        return tool_function(**kwargs)

    def is_registered(self, tool_name: str) -> bool:
        """Return whether a tool has been registered."""

        return (
            isinstance(tool_name, str)
            and tool_name in self._tools
        )

    @property
    def registered_tools(self) -> MappingProxyType:
        """Expose a read-only view of registered tool names."""

        return MappingProxyType(self._tools.copy())
