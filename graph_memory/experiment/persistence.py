from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TypeAlias, TypeVar

from omegaconf import OmegaConf
from pydantic import BaseModel, JsonValue

T = TypeVar("T", bound=BaseModel)
YamlValue: TypeAlias = JsonValue


def write_yaml_atomic(path: Path, value: object) -> None:
    primitive = _primitive(value)
    text = OmegaConf.to_yaml(
        OmegaConf.create(primitive),  # pyright: ignore[reportCallIssue, reportArgumentType]
        resolve=True,
        sort_keys=False,
    )
    _write_text_atomic(path, text)


def read_yaml(path: Path) -> YamlValue:
    return _yaml_value(
        OmegaConf.to_container(
            OmegaConf.load(path),
            resolve=True,
            throw_on_missing=True,
            enum_to_str=True,
        )
    )


def read_yaml_model(path: Path, model: type[T]) -> T:
    return model.model_validate(read_yaml(path))


def as_yaml_object(value: object) -> dict[str, YamlValue]:
    converted = _yaml_value(value)
    if not isinstance(converted, dict):
        raise TypeError("expected a YAML object")
    return converted


def _primitive(value: object) -> YamlValue:
    if isinstance(value, BaseModel):
        return _yaml_value(value.model_dump(mode="json", by_alias=True))
    if isinstance(value, tuple):
        return [_primitive(item) for item in value]
    if isinstance(value, list):
        return [_primitive(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _primitive(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value is not YAML serializable: {type(value).__name__}")


def _yaml_value(value: object) -> YamlValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_yaml_value(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("YAML object keys must be strings")
        return {str(key): _yaml_value(item) for key, item in value.items()}
    raise TypeError(f"invalid YAML value: {type(value).__name__}")


def _write_text_atomic(path: Path, text: str) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "read_yaml",
    "read_yaml_model",
    "as_yaml_object",
    "write_yaml_atomic",
]
