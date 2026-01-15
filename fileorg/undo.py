from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fileorg import paths


def _undo_path() -> Path:
    return paths.ensure_app_dir() / "last_action.json"


def save_last_move(applied_moves: list[dict[str, str]], scan_root: Path) -> None:
    """Persist the last applied move plan for a single-step undo."""
    data = {
        "action": "move",
        "scan_root": str(scan_root),
        "moves": applied_moves,
    }
    path = _undo_path()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_last_action() -> dict[str, Any] | None:
    path = _undo_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def clear_last_action() -> None:
    path = _undo_path()
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


def build_undo_plan(action: dict[str, Any]) -> list[dict[str, str]]:
    if action.get("action") != "move":
        return []
    moves = action.get("moves") or []
    undo_plan: list[dict[str, str]] = []
    for entry in reversed(moves):
        src = entry.get("src")
        dest = entry.get("dest")
        if isinstance(src, str) and isinstance(dest, str):
            undo_plan.append({"src": dest, "dest": src})
    return undo_plan
