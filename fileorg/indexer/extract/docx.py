from __future__ import annotations

from pathlib import Path

from docx import Document


def extract_docx_text(path: Path) -> str:
    doc = Document(path)
    parts = [paragraph.text for paragraph in doc.paragraphs if paragraph.text]
    return "\n".join(parts)
