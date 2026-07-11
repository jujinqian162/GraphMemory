from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel


CommandModel = TypeVar("CommandModel", bound=BaseModel)


def parse_command(
    model: type[CommandModel],
    arguments: Sequence[str],
) -> CommandModel:
    values: dict[str, str] = {}
    for argument in arguments:
        key, separator, value = argument.partition("=")
        if not separator or not key or key in values:
            raise ValueError(f"expected one unique key=value argument, got {argument!r}")
        values[key] = value
    return model.model_validate(values)


__all__ = ["parse_command"]

