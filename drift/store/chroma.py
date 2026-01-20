from __future__ import annotations

import os
import os
from dataclasses import dataclass
from pathlib import Path
from difflib import SequenceMatcher
from typing import Iterable
import re
from typing import TYPE_CHECKING, Optional

import chromadb
from chromadb.config import Settings

if TYPE_CHECKING:  # pragma: no cover
    from drift.store.meta import MetadataStore
# Hard-disable Chroma telemetry/posthog to avoid noisy errors in environments without network.
os.environ.setdefault("CHROMA_TELEMETRY", "0")
try:
    import posthog  # type: ignore

    posthog.capture = lambda *args, **kwargs: None  # type: ignore
    posthog.disabled = True  # type: ignore
except Exception:
    pass

# Hard-disable Chroma telemetry/posthog to avoid noisy errors in environments without network.
os.environ.setdefault("CHROMA_TELEMETRY", "0")
try:
    import posthog  # type: ignore

    posthog.capture = lambda *args, **kwargs: None  # type: ignore
    posthog.disabled = True  # type: ignore
except Exception:
    pass


@dataclass(frozen=True)
class SearchResult:
    path: str
    document: str
    distance: float


class ChromaStore:
    def __init__(self, base_dir: Path) -> None:
        base_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(base_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._text = self._client.get_or_create_collection("drift_text")
        self._image = self._client.get_or_create_collection("drift_image")

    def clear(self) -> None:
        self._client.delete_collection("drift_text")
        self._client.delete_collection("drift_image")
        self._text = self._client.get_or_create_collection("drift_text")
        self._image = self._client.get_or_create_collection("drift_image")

    def delete_path(self, path: str) -> None:
        self._text.delete(where={"path": path})
        self._image.delete(where={"path": path})

    def add_text_embeddings(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        if ids:
            self._text.add(
                ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
            )

    def add_image_embeddings(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        if ids:
            self._image.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def query_text(
        self,
        embedding: list[float],
        limit: int = 5,
        query: str | None = None,
        fetch_k: int | None = None,
        metadata: Optional["MetadataStore"] = None,
    ) -> list[SearchResult]:
        def _filename_score(q: str, path: str) -> float:
            if not q:
                return 0.0
            name = Path(path).name.lower()
            q_lower = q.lower().strip()
            tokens = [t for t in re.findall(r"[a-z0-9]+", q_lower) if t]
            exact_hits = sum(1 for t in tokens if t in name)
            prefix_bonus = 2.0 if name.startswith(q_lower) else 0.0
            fuzzy = SequenceMatcher(None, q_lower, name).ratio()
            # Heavy weight on direct token hits to prioritize filename matches.
            return exact_hits * 10 + prefix_bonus + fuzzy

        fetch_limit = limit
        if query:
            # Pull a larger candidate set so we can rerank by filename relevance.
            fetch_limit = max(fetch_k or (limit * 4), limit, 20)
        result = self._text.query(
            query_embeddings=[embedding],
            n_results=fetch_limit,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0] or []
        metadatas = result.get("metadatas", [[]])[0] or []
        distances = result.get("distances", [[]])[0] or []
        combined: dict[str, SearchResult] = {}
        for document, metadata_row, distance in zip(documents, metadatas, distances):
            path = metadata_row.get("path", "")
            combined[path] = SearchResult(
                path=path,
                document=document or "",
                distance=distance,
            )

        name_scores: dict[str, float] = {}
        if query and metadata is not None:
            for path in metadata.list_paths():
                score = _filename_score(query, path)
                if score > 0:
                    name_scores[path] = score
            # Keep top name hits only
            top_name_hits = sorted(
                name_scores.items(), key=lambda kv: kv[1], reverse=True
            )[:fetch_limit]
            name_scores = dict(top_name_hits)

            # Pull snippets for name-only hits not already present
            for path in name_scores:
                if path in combined:
                    continue
                doc = ""
                try:
                    snippet_result = self._text.get(
                        where={"path": path},
                        include=["documents"],
                        limit=1,
                    )
                    docs = snippet_result.get("documents") or []
                    if docs and docs[0]:
                        doc = docs[0][0] or ""
                except Exception:
                    doc = ""
                combined[path] = SearchResult(path=path, document=doc, distance=1.0)

        results = list(combined.values())
        if query:
            results.sort(
                key=lambda r: (-name_scores.get(r.path, 0.0), r.distance)
            )
        else:
            results.sort(key=lambda r: r.distance)
        return results[:limit]

    def file_embeddings(self) -> dict[str, list[float]]:
        """Average chunk embeddings per path across text and image collections."""
        aggregates: dict[str, list[float]] = {}
        counts: dict[str, int] = {}

        def add_batch(collection):
            data = collection.get(include=["embeddings", "metadatas"], limit=None)
            for emb, meta in zip(data.get("embeddings", []), data.get("metadatas", [])):
                path = meta.get("path")
                if not path or emb is None:
                    continue
                if path not in aggregates:
                    aggregates[path] = [0.0 for _ in emb]
                    counts[path] = 0
                aggregates[path] = [a + b for a, b in zip(aggregates[path], emb)]
                counts[path] += 1

        add_batch(self._text)
        add_batch(self._image)

        averaged: dict[str, list[float]] = {}
        for path, total in aggregates.items():
            cnt = counts[path]
            if cnt == 0:
                continue
            averaged[path] = [x / cnt for x in total]
        return averaged
