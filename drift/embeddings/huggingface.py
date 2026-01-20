from __future__ import annotations

from typing import Sequence

from sentence_transformers import SentenceTransformer

from drift.embeddings.base import TextEmbedder


class HuggingFaceTextEmbedder(TextEmbedder):
    def __init__(self, model: str, device: str | None = None) -> None:
        super().__init__(model)
        self._model = SentenceTransformer(model, device=device)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        # SentenceTransformer returns numpy arrays; convert to lists for Chroma.
        vectors = self._model.encode(
            list(texts), convert_to_numpy=True, normalize_embeddings=True, batch_size=16
        )
        return vectors.tolist()
