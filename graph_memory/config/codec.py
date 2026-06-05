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

    def write(self, path: str | Path, data: dict[str, Any]) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(data, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")


__all__ = ["JsonConfigCodec"]
