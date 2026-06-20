from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import cast

import pytest

from graph_memory.datasets.hotpotqa.projectors import HotpotQAToTemporalMemoryRankingRequest
from graph_memory.datasets.hotpotqa.records import HotpotQARankingRecord
from graph_memory.retrieval.methods.memory_stream.artifact import importance_content_digest
from graph_memory.retrieval.methods.memory_stream.contracts import (
    ImportanceArtifact,
    TaskImportanceRecord,
)
from graph_memory.retrieval.requests import TemporalMemoryRankingRequest
from graph_memory.validation import (
    ContractValidationError,
    select_importance_records,
    validate_importance_artifact,
)


def _task(task_id: str = "hotpot_ms_1") -> HotpotQARankingRecord:
    return {
        "task_id": task_id,
        "question": "Which river runs through Paris?",
        "candidate_sentences": [
            {
                "sentence_id": "m0",
                "text": "The Eiffel Tower is in Paris.",
                "title": "Eiffel Tower",
                "sentence_index": 0,
                "position": 0,
            },
            {
                "sentence_id": "m1",
                "text": "The Seine runs through Paris.",
                "title": "Paris",
                "sentence_index": 0,
                "position": 1,
            },
        ],
    }


def _request(task: HotpotQARankingRecord) -> TemporalMemoryRankingRequest:
    return HotpotQAToTemporalMemoryRankingRequest().project(task, {})


def _artifact(requests: list[TemporalMemoryRankingRequest]) -> ImportanceArtifact:
    return {
        "schema_version": 1,
        "method": "memory_stream",
        "tasks": [
            {
                "task_id": request.task_id,
                "content_digest": importance_content_digest(request),
                "scores": {candidate.item_id: index + 1 for index, candidate in enumerate(request.candidates)},
            }
            for request in requests
        ],
    }


def test_hotpotqa_temporal_projection_preserves_legacy_importance_digest_semantics() -> None:
    request = _request(_task())
    payload = {
        "items": [
            {
                "id": "m0",
                "source": "Eiffel Tower",
                "text": "The Eiffel Tower is in Paris.",
                "position": 0,
            },
            {
                "id": "m1",
                "source": "Paris",
                "text": "The Seine runs through Paris.",
                "position": 1,
            },
        ]
    }
    expected_digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    assert request.candidates[0].text == "The Eiffel Tower is in Paris."
    assert importance_content_digest(request) == expected_digest


def test_importance_artifact_accepts_only_compact_schema() -> None:
    requests = [_request(_task())]
    artifact = _artifact(requests)

    validate_importance_artifact(artifact, requests)

    legacy = {
        **artifact,
        "model": "gpt-5.4-mini",
        "prompt_version": "memory-stream-importance-v2",
        "generation": {"do_sample": False, "use_cache": True, "max_new_tokens": 2048},
    }
    with pytest.raises(ContractValidationError, match="unknown fields"):
        validate_importance_artifact(legacy, requests)


def test_importance_artifact_requires_order_digest_and_exact_node_coverage() -> None:
    requests = [_request(_task("hotpot_ms_1")), _request(_task("hotpot_ms_2"))]
    artifact = _artifact(requests)

    out_of_order = {**artifact, "tasks": list(reversed(artifact["tasks"]))}
    with pytest.raises(ContractValidationError, match="order"):
        validate_importance_artifact(out_of_order, requests)

    bad_digest = deepcopy(artifact)
    bad_digest["tasks"][0]["content_digest"] = "bad"
    with pytest.raises(ContractValidationError, match="content_digest"):
        validate_importance_artifact(bad_digest, requests)

    missing_node = deepcopy(artifact)
    del missing_node["tasks"][0]["scores"]["m1"]
    with pytest.raises(ContractValidationError, match="missing=.*m1"):
        validate_importance_artifact(missing_node, requests)


def test_importance_artifact_rejects_invalid_scores() -> None:
    requests = [_request(_task())]
    artifact = _artifact(requests)

    for invalid in (True, 4.5, 0, 11):
        changed = deepcopy(artifact)
        scores = cast(dict[str, object], changed["tasks"][0]["scores"])
        scores["m0"] = invalid
        with pytest.raises(ContractValidationError, match="integer|1-10"):
            validate_importance_artifact(changed, requests)


def test_global_importance_artifact_selects_subset_in_requested_order() -> None:
    requests = [_request(_task("hotpot_ms_1")), _request(_task("hotpot_ms_2"))]
    artifact = _artifact(requests)

    selected = select_importance_records(artifact, [requests[1], requests[0]])

    assert [record["task_id"] for record in selected] == ["hotpot_ms_2", "hotpot_ms_1"]


def test_global_importance_artifact_rejects_missing_duplicate_and_stale_records() -> None:
    records = [_task("hotpot_ms_1"), _task("hotpot_ms_2")]
    requests = [_request(record) for record in records]
    artifact = _artifact(requests)

    with pytest.raises(ContractValidationError, match="missing task_id=hotpot_ms_2"):
        _ = select_importance_records({**artifact, "tasks": artifact["tasks"][:1]}, [requests[1]])

    duplicate_record: TaskImportanceRecord = artifact["tasks"][0]
    with pytest.raises(ContractValidationError, match="duplicate task_id=hotpot_ms_1"):
        _ = select_importance_records(
            {**artifact, "tasks": [duplicate_record, duplicate_record]},
            [requests[0]],
        )

    stale_record = deepcopy(records[0])
    stale_record["candidate_sentences"][0]["text"] = "Changed content."
    with pytest.raises(ContractValidationError, match="content_digest"):
        _ = select_importance_records(artifact, [_request(stale_record)])
