from __future__ import annotations

import json
from collections.abc import Sequence

from pydantic import BaseModel

from graph_memory.datasets.twowiki_provenance import (
    convert_twowiki_source_records,
    resolve_worker_count,
)


def _source_example(raw_id: str) -> dict[str, object]:
    return {
        "_id": raw_id,
        "type": "compositional",
        "question": "In which country is the birthplace of Alpha located?",
        "context": [
            ["Alpha", ["Alpha was born in Bridge City.", "Alpha is an artist."]],
            [
                "Bridge City",
                ["Bridge City is located in Country Z.", "It has a river."],
            ],
            ["Wrong One", ["Wrong One is located elsewhere."]],
            ["Wrong Two", ["Wrong Two mentions Alpha but not the answer."]],
        ],
        "supporting_facts": [["Alpha", 0], ["Bridge City", 0]],
        "evidences": [
            ["Alpha", "birth place", "Bridge City"],
            ["Bridge City", "country", "Country Z"],
        ],
        "evidences_id": [],
        "answer_id": "Country Z",
        "answer": "Country Z",
    }


def _canonical(records: Sequence[object]) -> str:
    return json.dumps(
        [
            record.model_dump(mode="json")
            if isinstance(record, BaseModel)
            else record
            for record in records
        ],
        sort_keys=True,
        separators=(",", ":"),
    )


def test_parallel_bm25_output_is_byte_identical_to_serial() -> None:
    # Enough records that a 4-way split gives each worker more than one record.
    raw = [_source_example(f"rec-{index}") for index in range(12)]

    serial = convert_twowiki_source_records(raw, candidate_cap=6, seed=17, workers=1)
    parallel = convert_twowiki_source_records(raw, candidate_cap=6, seed=17, workers=4)

    assert _canonical(serial.records) == _canonical(parallel.records)
    assert serial.rejected_reason_counts == parallel.rejected_reason_counts
    assert len(serial.records) == 12


def test_resolve_worker_count_defaults_and_floor() -> None:
    assert resolve_worker_count(4) == 4
    assert resolve_worker_count(0) == 1
    assert resolve_worker_count(None) >= 1
