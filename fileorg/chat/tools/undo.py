from __future__ import annotations

from fileorg.chat.tools.common import ToolResult, to_json
from fileorg.undo import build_undo_plan, load_last_action


def tool_undo_last_action() -> ToolResult:
    action = load_last_action()
    if not action or action.get("action") != "move":
        return ToolResult("undo_last_action", to_json({"error": "No move action available to undo."}))

    plan = build_undo_plan(action)
    if not plan:
        return ToolResult("undo_last_action", to_json({"error": "Could not build undo plan."}))

    return ToolResult("undo_last_action", to_json({"action": "move", "plan": plan, "scan_root": action.get("scan_root")}))
