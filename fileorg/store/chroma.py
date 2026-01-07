from __future__ import annotations

import os
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.config import Settings

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
        self._text = self._client.get_or_create_collection("fileorg_text")
        self._image = self._client.get_or_create_collection("fileorg_image")

    def clear(self) -> None:
        self._client.delete_collection("fileorg_text")
        self._client.delete_collection("fileorg_image")
        self._text = self._client.get_or_create_collection("fileorg_text")
        self._image = self._client.get_or_create_collection("fileorg_image")

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

    def query_text(self, embedding: list[float], limit: int = 5) -> list[SearchResult]:
        result = self._text.query(
            query_embeddings=[embedding],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )
        if not result["ids"] or not result["ids"][0]:
            return []
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        results: list[SearchResult] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            results.append(
                SearchResult(
                    path=metadata.get("path", ""),
                    document=document or "",
                    distance=distance,
                )
            )
        return results

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
