from __future__ import annotations

import csv
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeAlias

from pydantic import BaseModel

JsonDict: TypeAlias = dict[str, Any]


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        file.write("\n")


def write_json_atomic(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temp_path = Path(file.name)
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temp_path.replace(output_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: str | Path, rows: list[Any], fieldnames: list[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                row.model_dump(mode="json", by_alias=True)
                if isinstance(row, BaseModel)
                else row
            )


def write_jsonl(path: str | Path, records: list[Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=_json_default,
                )
            )
            file.write("\n")


def _json_default(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def merge_config(
    defaults: JsonDict,
    config_file: JsonDict | None = None,
    cli_overrides: JsonDict | None = None,
) -> JsonDict:
    effective_config = deepcopy(defaults)
    if config_file:
        effective_config = _deep_merge(effective_config, config_file)
    if cli_overrides:
        effective_config = _deep_merge(effective_config, cli_overrides)
    return effective_config


def _deep_merge(base: JsonDict, override: JsonDict) -> JsonDict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


__all__ = [
    "merge_config",
    "read_csv",
    "read_json",
    "write_csv",
    "write_json",
    "write_json_atomic",
    "write_jsonl",
]
