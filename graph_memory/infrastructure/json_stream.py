from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import cast


def iter_json_array(
    path: str | Path,
    *,
    chunk_size: int = 1 << 20,
) -> Iterator[object]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    decoder = json.JSONDecoder()
    with Path(path).open("r", encoding="utf-8") as file:
        buffer = ""
        position = 0
        exhausted = False

        def refill() -> None:
            nonlocal buffer, position, exhausted
            if position:
                buffer = buffer[position:]
                position = 0
            chunk = file.read(chunk_size)
            if chunk:
                buffer += chunk
            else:
                exhausted = True

        refill()
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                break
            if exhausted:
                raise ValueError(f"{path} is empty")
            refill()
        if buffer[position] != "[":
            raise ValueError(f"{path} must contain a top-level JSON array")
        position += 1

        while True:
            while True:
                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                if position < len(buffer) or exhausted:
                    break
                refill()
            if position < len(buffer) and buffer[position] == "]":
                return
            if exhausted and position >= len(buffer):
                raise ValueError(f"unterminated JSON array in {path}")
            try:
                value, end = cast(
                    tuple[object, int], decoder.raw_decode(buffer, position)
                )
            except json.JSONDecodeError:
                if exhausted:
                    raise ValueError(f"invalid JSON array in {path}") from None
                refill()
                continue
            position = end
            yield value


__all__ = ["iter_json_array"]
