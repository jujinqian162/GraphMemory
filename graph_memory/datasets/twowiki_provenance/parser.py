from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from graph_memory.datasets.twowiki_provenance.records import (
    TWOWIKI_PROVENANCE_SCHEMA_VERSION,
    TwoWikiProvenanceRawRecord,
)

RAW_FIELDS = {"schema_version", "ranking", "label"}


def parse_twowiki_provenance_records(
    raw_records: Sequence[object],
) -> list[TwoWikiProvenanceRawRecord]:
    return [
        parse_twowiki_provenance_record(record, record_index=index)
        for index, record in enumerate(raw_records)
    ]


def parse_twowiki_provenance_record(
    value: object,
    *,
    record_index: int | None = None,
) -> TwoWikiProvenanceRawRecord:
    path = (
        "2Wiki provenance raw record"
        if record_index is None
        else f"2Wiki provenance raw record index={record_index}"
    )
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object.")
    unknown = sorted(set(value) - RAW_FIELDS)
    if unknown:
        raise ValueError(f"{path} contains unknown fields={unknown}.")
    if value.get("schema_version") != TWOWIKI_PROVENANCE_SCHEMA_VERSION:
        raise ValueError(
            f"{path} schema_version must be {TWOWIKI_PROVENANCE_SCHEMA_VERSION}."
        )
    if not isinstance(value.get("ranking"), Mapping):
        raise ValueError(f"{path} ranking must be an object.")
    if not isinstance(value.get("label"), Mapping):
        raise ValueError(f"{path} label must be an object.")
    return cast(TwoWikiProvenanceRawRecord, cast(object, dict(value)))


__all__ = [
    "parse_twowiki_provenance_record",
    "parse_twowiki_provenance_records",
]
