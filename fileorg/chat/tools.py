from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fileorg.store import ChromaStore, MetadataStore


@dataclass
class ToolResult:
    name: str
    content: str


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def tool_search(query: str, limit: int, chroma: ChromaStore, embedder) -> ToolResult:
    embedding = embedder.embed_query(query)
    results = chroma.query_text(embedding, limit=limit)
    payload = [
        {"path": r.path, "distance": r.distance, "snippet": r.document[:400]}
        for r in results
    ]
    return ToolResult("search_files", _json(payload))


def tool_duplicates(metadata: MetadataStore) -> ToolResult:
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for rec in metadata.list_records():
        if rec.content_hash:
            by_hash[rec.content_hash].append(
                {"path": rec.path, "size": rec.size, "mtime": rec.mtime}
            )
    dupes = [group for group in by_hash.values() if len(group) > 1]
    return ToolResult("find_duplicates", _json(dupes))


def tool_stale(days: int, metadata: MetadataStore) -> ToolResult:
    cutoff = time.time() - days * 86400
    stale = []
    for rec in metadata.list_records():
        if rec.mtime < cutoff:
            stale.append({"path": rec.path, "mtime": rec.mtime, "size": rec.size})
    stale_sorted = sorted(stale, key=lambda x: x["mtime"])
    return ToolResult("find_stale_files", _json(stale_sorted[:200]))


def tool_suggest_structure(
    chroma: ChromaStore,
    client,
    min_cluster_size: int = 3,
    min_samples: int = 2,
) -> ToolResult:
    try:
        import hdbscan  # type: ignore
    except ImportError:
        return ToolResult(
            "suggest_structure",
            "hdbscan not installed; cannot cluster. Install dependencies and re-run.",
        )

    embeddings = chroma.file_embeddings()
    if not embeddings:
        return ToolResult(
            "suggest_structure",
            "No embeddings available. Run indexing first.",
        )

    lengths: dict[int, list[str]] = defaultdict(list)
    for path, vec in embeddings.items():
        lengths[len(vec)].append(path)
    if not lengths:
        return ToolResult(
            "suggest_structure",
            "No usable embeddings found.",
        )

    suggestions = []
    cluster_idx = 0
    for dim, paths in lengths.items():
        if len(paths) < min_cluster_size:
            continue
        vectors = np.stack([embeddings[p] for p in paths])
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=0.5,
        )
        labels = clusterer.fit_predict(vectors)
        clusters: dict[int, list[str]] = defaultdict(list)
        for path, label in zip(paths, labels):
            if label == -1:
                continue
            clusters[int(label)].append(path)
        for label, c_paths in clusters.items():
            basenames = [Path(p).name for p in c_paths]
            sample = ", ".join(basenames[:10])
            prompt = (
                "Name a concise folder (max 30 chars) for these related files:\n"
                f"{sample}\nFolder name:"
            )
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=20,
                    temperature=0.2,
                )
                name = (resp.choices[0].message.content or "").strip().splitlines()[0]
            except Exception:
                name = ""
            if not name:
                name = f"Cluster {cluster_idx}"
            suggestions.append(
                {
                    "folder": name,
                    "count": len(c_paths),
                    "sample_files": basenames[:5],
                }
            )
            cluster_idx += 1

    if not suggestions:
        return ToolResult(
            "suggest_structure",
            "No meaningful clusters found.",
        )
    return ToolResult("suggest_structure", _json(suggestions))


def tool_preview_moves(plan: list[dict]) -> ToolResult:
    # The model can propose a plan; we just echo it back for now.
    return ToolResult("preview_moves", _json(plan))


def tool_definitions() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search_files",
                "description": "Semantic search over indexed text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_duplicates",
                "description": "Find duplicate files by exact content hash",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_stale_files",
                "description": "List stale files untouched for N days",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "days": {"type": "integer", "default": 180},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "suggest_structure",
                "description": "Suggest a folder structure based on clustered embeddings",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "min_cluster_size": {"type": "integer", "default": 3},
                        "min_samples": {"type": "integer", "default": 2},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "preview_moves",
                "description": "Preview file move/rename plan before execution",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plan": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "src": {"type": "string"},
                                    "dest": {"type": "string"},
                                },
                                "required": ["src", "dest"],
                            },
                        }
                    },
                    "required": ["plan"],
                },
            },
        },
    ]
