from __future__ import annotations

from fileorg.config import AppConfig
from fileorg.embeddings.base import ImageEmbedder, TextEmbedder
from fileorg.embeddings.huggingface import HuggingFaceTextEmbedder
from fileorg.embeddings.image import ClipImageEmbedder
from fileorg.embeddings.text import OllamaTextEmbedder


def build_text_embedder(config: AppConfig) -> TextEmbedder:
    provider = config.embeddings.text_provider.lower()
    if provider == "ollama":
        return OllamaTextEmbedder(config.embeddings.text_model)
    if provider == "huggingface":
        return HuggingFaceTextEmbedder(config.embeddings.text_model)
    raise ValueError(f"Unsupported text embedding provider: {provider}")


def build_image_embedder(config: AppConfig) -> ImageEmbedder:
    provider = config.embeddings.image_provider.lower()
    if provider == "clip":
        return ClipImageEmbedder(config.embeddings.image_model)
    raise ValueError(f"Unsupported image embedding provider: {provider}")
