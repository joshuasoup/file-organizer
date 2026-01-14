from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def parse_structure_suggestions(content: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(content)
    except Exception:
        return []

    suggestions: list[Any]
    if isinstance(data, dict) and "suggestions" in data:
        suggestions = data.get("suggestions") or []
    elif isinstance(data, list):
        suggestions = data
    else:
        return []

    normalized: list[dict[str, Any]] = []
    for item in suggestions:
        if not isinstance(item, dict):
            continue
        folder = str(item.get("folder") or "Unknown")
        path_value = str(item.get("path") or folder)
        try:
            count = int(item.get("count", 0))
        except Exception:
            count = 0
        sample = item.get("sample_files") or []
        paths = item.get("paths") or []
        normalized.append(
            {
                "folder": folder,
                "path": path_value,
                "count": count,
                "sample_files": [str(s) for s in sample],
                "paths": [str(p) for p in paths],
            }
        )
    return normalized


def build_move_plan(suggestions: list[dict[str, Any]], scan_root: Path | None = None) -> list[dict[str, str]]:
    plan: list[dict[str, str]] = []
    for suggestion in suggestions:
        dest_folder = suggestion.get("path") or suggestion.get("folder")
        if not dest_folder:
            continue
        dest_base = Path(dest_folder)
        if scan_root is not None and not dest_base.is_absolute():
            dest_base = scan_root / dest_base
        paths = suggestion.get("paths") or []
        for src in paths:
            src_path = Path(src)
            dest = dest_base / src_path.name
            plan.append({"src": str(src_path), "dest": str(dest)})
    return plan


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def apply_move_plan(plan: list[dict[str, str]], scan_root: Path) -> dict[str, int]:
    root = scan_root.resolve()
    stats = {
        "moved": 0,
        "skipped_missing": 0,
        "skipped_outside": 0,
        "skipped_conflict": 0,
        "errors": 0,
    }
    for entry in plan:
        raw_src = entry.get("src", "")
        raw_dest = entry.get("dest", "")
        src = Path(raw_src)
        dest = Path(raw_dest)
        dest_dir_hint = raw_dest.endswith("/") or raw_dest in {"", ".", "./"}
        if not src.is_absolute():
            src = root / src
        if not dest.is_absolute():
            dest = root / dest

        try:
            src = src.resolve()
            dest = dest.resolve()
        except Exception:
            stats["errors"] += 1
            continue

        if dest_dir_hint or dest.is_dir() or (not dest.exists() and dest.suffix == ""):
            dest = dest / src.name

        if not _is_within(src, root) or not _is_within(dest, root):
            stats["skipped_outside"] += 1
            continue
        if not src.exists():
            stats["skipped_missing"] += 1
            continue
        if dest.exists():
            stats["skipped_conflict"] += 1
            continue
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dest)
            stats["moved"] += 1
        except Exception:
            stats["errors"] += 1
    return stats


def parse_move_plan(content: str) -> list[dict[str, str]]:
    try:
        data = json.loads(content)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    plan: list[dict[str, str]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        src = entry.get("src")
        dest = entry.get("dest")
        if isinstance(src, str) and isinstance(dest, str):
            plan.append({"src": src, "dest": dest})
    return plan


def apply_delete_plan(items: list[str], scan_root: Path) -> dict[str, int]:
    root = scan_root.resolve()
    stats = {
        "deleted": 0,
        "skipped_missing": 0,
        "skipped_outside": 0,
        "errors": 0,
    }
    for raw in items:
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        try:
            path = path.resolve()
        except Exception:
            stats["errors"] += 1
            continue
        if not _is_within(path, root):
            stats["skipped_outside"] += 1
            continue
        if not path.exists():
            stats["skipped_missing"] += 1
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            stats["deleted"] += 1
        except Exception:
            stats["errors"] += 1
    return stats
