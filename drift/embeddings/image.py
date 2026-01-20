from __future__ import annotations

from typing import Iterable

from PIL import Image

from drift.embeddings.base import ImageEmbedder


class ClipImageEmbedder(ImageEmbedder):
    def __init__(self, model: str, device: str | None = None) -> None:
        super().__init__(model)
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "CLIP dependencies missing. Install open_clip_torch and torch."
            ) from exc

        model_name = model.replace("/", "-")
        pretrained_tag = "openai"
        force_quick_gelu = False
        # OpenAI weights expect QuickGELU; prefer the matching config to avoid warnings.
        if "quickgelu" not in model_name.lower() and pretrained_tag == "openai":
            quick_model_name = f"{model_name}-quickgelu"
            if open_clip.get_model_config(quick_model_name):
                model_name = quick_model_name
            else:
                force_quick_gelu = True

        self._torch = torch
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_tag, force_quick_gelu=force_quick_gelu
        )
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()

    def embed(self, images: Iterable[Image.Image]) -> list[list[float]]:
        torch = self._torch
        tensors = [self._preprocess(image) for image in images]
        if not tensors:
            return []
        batch = torch.stack(tensors).to(self._device)
        with torch.no_grad():
            vectors = self._model.encode_image(batch)
            vectors = vectors / vectors.norm(dim=-1, keepdim=True)
        return vectors.cpu().tolist()
