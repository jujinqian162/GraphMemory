from __future__ import annotations

from typing import TypeVar

from graph_memory.compat import StrEnum


class StageId(StrEnum):
    PREPARE = "prepare"
    GRAPHS = "graphs"
    PAIRS = "pairs"
    TUNE = "tune"
    TRAIN = "train"
    RETRIEVE = "retrieve"
    EVALUATE = "evaluate"
    AGGREGATE = "aggregate"
    EXPERIMENT_INIT = "experiment_init"
    EXPERIMENT_PLAN = "experiment_plan"


EnumValue = TypeVar("EnumValue", bound=StrEnum)


def parse_closed_value(enum_type: type[EnumValue], value: str, *, label: str) -> EnumValue:
    try:
        return enum_type(value)
    except ValueError as error:
        allowed = ", ".join(member.value for member in enum_type)
        raise ValueError(f"Unknown {label}={value!r}; allowed values: {allowed}") from error


__all__ = ["StageId", "StrEnum", "parse_closed_value"]
