from __future__ import annotations

from collections import defaultdict

from drift.chat.tools.common import ToolResult, to_json
from drift.store import MetadataStore


def tool_duplicates(metadata: MetadataStore) -> ToolResult:
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for rec in metadata.list_records():
        if rec.content_hash:
            by_hash[rec.content_hash].append(
                {"path": rec.path, "size": rec.size, "mtime": rec.mtime}
            )
    dupes = [group for group in by_hash.values() if len(group) > 1]
    return ToolResult("find_duplicates", to_json(dupes))
