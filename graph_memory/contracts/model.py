from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    StringConstraints,
)

JsonObject: TypeAlias = Mapping[str, "JsonValue"]
JsonArray: TypeAlias = Sequence["JsonValue"]
JsonValue: TypeAlias = str | int | float | bool | None | JsonArray | JsonObject

NonEmptyStr = Annotated[StrictStr, StringConstraints(min_length=1)]
FiniteFloat = Annotated[StrictFloat, Field(allow_inf_nan=False)]
NonNegativeFiniteFloat = Annotated[
    StrictFloat, Field(allow_inf_nan=False, ge=0.0)
]
PositiveFiniteFloat = Annotated[StrictFloat, Field(allow_inf_nan=False, gt=0.0)]
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
PositiveInt = Annotated[StrictInt, Field(gt=0)]

FORBIDDEN_LABEL_FIELDS = frozenset(
    {
        "answer",
        "answers",
        "decomposition",
        "dependency_edges",
        "evidence_triples",
        "gold_evidence_item_ids",
        "gold_paragraph_ids",
        "gold_sentence_ids",
        "gold_supporting_facts",
        "supporting_facts",
    }
)


class DomainModel(BaseModel):
    """Shared mechanics for closed immutable domain contracts.

    Field schemas and scientific invariants belong to concrete domain models;
    this base intentionally owns no artifact fields.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        populate_by_name=True,
    )

    def __init__(self, *args: object, **data: Any) -> None:
        """Bind positional values through Pydantic's declared field order.

        This preserves the value-object constructor style used by the domain
        while all coercion, rejection, and invariant checking remains owned by
        Pydantic.
        """

        if args:
            field_names = tuple(type(self).model_fields)
            if len(args) > len(field_names):
                raise TypeError(
                    f"{type(self).__name__} accepts at most {len(field_names)} "
                    f"positional arguments; got {len(args)}"
                )
            duplicates = set(field_names[: len(args)]).intersection(data)
            if duplicates:
                names = ", ".join(sorted(duplicates))
                raise TypeError(f"multiple values for fields: {names}")
            data = dict(zip(field_names[: len(args)], args, strict=True)) | data
        super().__init__(**data)


def reject_label_fields(value: object, *, path: str = "artifact") -> None:
    """Reject supervised labels recursively from model-input artifacts."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}"
            if key.lower() in FORBIDDEN_LABEL_FIELDS:
                raise ValueError(f"label field is forbidden at {child_path}")
            reject_label_fields(child, path=child_path)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            reject_label_fields(child, path=f"{path}[{index}]")


__all__ = [
    "DomainModel",
    "FORBIDDEN_LABEL_FIELDS",
    "FiniteFloat",
    "JsonArray",
    "JsonObject",
    "JsonValue",
    "NonEmptyStr",
    "NonNegativeFiniteFloat",
    "NonNegativeInt",
    "PositiveFiniteFloat",
    "PositiveInt",
    "StrictBool",
    "reject_label_fields",
]
