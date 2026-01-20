from __future__ import annotations

from typing import Iterable, Sequence

from PIL import Image


class TextEmbedder:
    def __init__(self, model: str) -> None:
        self.model = model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class ImageEmbedder:
    def __init__(self, model: str) -> None:
        self.model = model

    def embed(self, images: Iterable[Image.Image]) -> list[list[float]]:
        raise NotImplementedError
