from __future__ import annotations

from pathlib import Path

from PIL import Image


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()
