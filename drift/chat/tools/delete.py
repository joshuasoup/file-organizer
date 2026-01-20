from __future__ import annotations

from drift.chat.tools.common import ToolResult, to_json


def tool_delete(items: list[str]) -> ToolResult:
    """Preview a delete plan."""
    return ToolResult("delete_items", to_json(items))
