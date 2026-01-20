from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Sequence

from drift.embeddings.base import TextEmbedder


def _normalize_base_url(value: str) -> str:
    if value.startswith("http://") or value.startswith("https://"):
        return value.rstrip("/")
    return f"http://{value.rstrip('/')}"


class OllamaTextEmbedder(TextEmbedder):
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        timeout: int = 60,
    ) -> None:
        super().__init__(model)
        raw_url = base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.base_url = _normalize_base_url(raw_url)
        self.timeout = timeout

    def _request(self, payload: dict) -> dict:
        url = f"{self.base_url}/api/embeddings"
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError("Failed to reach Ollama for embeddings.") from exc

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            response = self._request({"model": self.model, "prompt": text})
            vector = response.get("embedding")
            if not vector:
                raise RuntimeError("Ollama returned no embedding data.")
            embeddings.append(vector)
        return embeddings
