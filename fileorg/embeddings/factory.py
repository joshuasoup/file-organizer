from __future__ import annotations

from threading import Lock

from fileorg.config import AppConfig
from fileorg.embeddings.base import ImageEmbedder, TextEmbedder
from fileorg.embeddings.huggingface import HuggingFaceTextEmbedder
from fileorg.embeddings.image import ClipImageEmbedder
from fileorg.embeddings.text import OllamaTextEmbedder

_TEXT_EMBEDDER_CACHE: dict[tuple[str, str], TextEmbedder] = {}
_TEXT_EMBEDDER_LOCK = Lock()


def build_text_embedder(config: AppConfig) -> TextEmbedder:
    provider = config.embeddings.text_provider.lower()
    key = (provider, config.embeddings.text_model)
    with _TEXT_EMBEDDER_LOCK:
        if key in _TEXT_EMBEDDER_CACHE:
            return _TEXT_EMBEDDER_CACHE[key]
        if provider == "ollama":
            embedder = OllamaTextEmbedder(config.embeddings.text_model)
        elif provider == "huggingface":
            embedder = HuggingFaceTextEmbedder(config.embeddings.text_model)
        else:
            raise ValueError(f"Unsupported text embedding provider: {provider}")
        _TEXT_EMBEDDER_CACHE[key] = embedder
        return embedder


def build_image_embedder(config: AppConfig) -> ImageEmbedder:
    provider = config.embeddings.image_provider.lower()
    if provider == "clip":
        return ClipImageEmbedder(config.embeddings.image_model)
    raise ValueError(f"Unsupported image embedding provider: {provider}")
