from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonConfigCodec:
    def read(self, path: str | Path) -> dict[str, Any]:
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError(f"Config must be a JSON object: {config_path}")
        return data

__all__ = ["JsonConfigCodec"]
