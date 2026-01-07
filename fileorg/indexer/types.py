from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class FileType(str, Enum):
    TEXT = "text"
    CODE = "code"
    PDF = "pdf"
    DOCX = "docx"
    IMAGE = "image"
    OTHER = "other"


@dataclass(frozen=True)
class FileInfo:
    path: Path
    size: int
    mtime: float
    file_type: FileType
