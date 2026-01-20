from __future__ import annotations

from drift.chat.tools.common import ToolResult, to_json


def tool_preview_moves(plan: list[dict]) -> ToolResult:
    return ToolResult("preview_moves", to_json(plan))


def tool_move(plan: list[dict]) -> ToolResult:
    """Alias for preview_moves, used for direct move requests."""
    return ToolResult("move_files", to_json(plan))
