from __future__ import annotations

from collections.abc import Sequence

from graph_memory.datasets.twowiki_provenance.records import (
    TwoWikiProvenanceRawRecord,
)


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
    try:
        return TwoWikiProvenanceRawRecord.model_validate(value)
    except ValueError as error:
        location = (
            "2Wiki provenance raw record"
            if record_index is None
            else f"2Wiki provenance raw record index={record_index}"
        )
        error.add_note(location)
        raise


__all__ = [
    "parse_twowiki_provenance_record",
    "parse_twowiki_provenance_records",
]
