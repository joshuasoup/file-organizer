from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    name: str
    content: str


def to_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)
