from __future__ import annotations

import re
from dataclasses import dataclass

TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class TextChunk:
    text: str
    start: int
    end: int


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
    max_chunks: int | None = None,
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    tokens = TOKEN_RE.findall(text)
    if not tokens:
        return []

    step = chunk_size - overlap
    chunks: list[TextChunk] = []
    for start in range(0, len(tokens), step):
        end = min(start + chunk_size, len(tokens))
        chunk_text_value = " ".join(tokens[start:end])
        chunks.append(TextChunk(text=chunk_text_value, start=start, end=end))
        if max_chunks is not None and len(chunks) >= max_chunks:
            break
        if end == len(tokens):
            break
    return chunks
