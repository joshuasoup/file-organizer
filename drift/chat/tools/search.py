from __future__ import annotations

from drift.chat.tools.common import ToolResult, to_json
from drift.store import ChromaStore


def tool_search(
    query: str,
    limit: int,
    chroma: ChromaStore,
    embedder,
    metadata=None,
) -> ToolResult:
    embedding = embedder.embed_query(query)
    results = chroma.query_text(
        embedding, limit=limit, query=query, metadata=metadata
    )
    payload = [
        {"path": r.path, "distance": r.distance, "snippet": r.document[:400]}
        for r in results
    ]
    return ToolResult("search_files", to_json(payload))
