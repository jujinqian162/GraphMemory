from __future__ import annotations

from copy import deepcopy
from typing import cast

import pytest

from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.retrieval.methods.memory_stream.artifact import importance_content_digest
from graph_memory.retrieval.methods.memory_stream.contracts import (
    ImportanceArtifact,
    TaskImportanceRecord,
)
from graph_memory.validation import (
    ContractValidationError,
    select_importance_records,
    validate_importance_artifact,
)


def _task(task_id: str = "hotpot_ms_1") -> MemoryTaskInput:
    return {
        "task_id": task_id,
        "query": "Which river runs through Paris?",
        "memory_items": [
            {
                "id": "m0",
                "node_type": "document_sentence",
                "text": "The Eiffel Tower is in Paris.",
                "source": "Eiffel Tower",
                "sentence_id": 0,
                "position": 0,
            },
            {
                "id": "m1",
                "node_type": "document_sentence",
                "text": "The Seine runs through Paris.",
                "source": "Paris",
                "sentence_id": 0,
                "position": 1,
            },
        ],
    }


def _artifact(tasks: list[MemoryTaskInput]) -> ImportanceArtifact:
    return {
        "schema_version": 1,
        "method": "memory_stream",
        "tasks": [
            {
                "task_id": task["task_id"],
                "content_digest": importance_content_digest(task),
                "scores": {item["id"]: index + 1 for index, item in enumerate(task["memory_items"])},
            }
            for task in tasks
        ],
    }


def test_importance_artifact_accepts_only_compact_schema() -> None:
    tasks = [_task()]
    artifact = _artifact(tasks)

    validate_importance_artifact(artifact, tasks)

    legacy = {
        **artifact,
        "model": "gpt-5.4-mini",
        "prompt_version": "memory-stream-importance-v2",
        "generation": {"do_sample": False, "use_cache": True, "max_new_tokens": 2048},
    }
    with pytest.raises(ContractValidationError, match="unknown fields"):
        validate_importance_artifact(legacy, tasks)


def test_importance_artifact_requires_order_digest_and_exact_node_coverage() -> None:
    tasks = [_task("hotpot_ms_1"), _task("hotpot_ms_2")]
    artifact = _artifact(tasks)

    out_of_order = {**artifact, "tasks": list(reversed(artifact["tasks"]))}
    with pytest.raises(ContractValidationError, match="order"):
        validate_importance_artifact(out_of_order, tasks)

    bad_digest = deepcopy(artifact)
    bad_digest["tasks"][0]["content_digest"] = "bad"
    with pytest.raises(ContractValidationError, match="content_digest"):
        validate_importance_artifact(bad_digest, tasks)

    missing_node = deepcopy(artifact)
    del missing_node["tasks"][0]["scores"]["m1"]
    with pytest.raises(ContractValidationError, match="missing=.*m1"):
        validate_importance_artifact(missing_node, tasks)


def test_importance_artifact_rejects_invalid_scores() -> None:
    tasks = [_task()]
    artifact = _artifact(tasks)

    for invalid in (True, 4.5, 0, 11):
        changed = deepcopy(artifact)
        scores = cast(dict[str, object], changed["tasks"][0]["scores"])
        scores["m0"] = invalid
        with pytest.raises(ContractValidationError, match="integer|1-10"):
            validate_importance_artifact(changed, tasks)


def test_global_importance_artifact_selects_subset_in_requested_order() -> None:
    tasks = [_task("hotpot_ms_1"), _task("hotpot_ms_2")]
    artifact = _artifact(tasks)

    selected = select_importance_records(artifact, [tasks[1], tasks[0]])

    assert [record["task_id"] for record in selected] == ["hotpot_ms_2", "hotpot_ms_1"]


def test_global_importance_artifact_rejects_missing_duplicate_and_stale_records() -> None:
    tasks = [_task("hotpot_ms_1"), _task("hotpot_ms_2")]
    artifact = _artifact(tasks)

    with pytest.raises(ContractValidationError, match="missing task_id=hotpot_ms_2"):
        _ = select_importance_records({**artifact, "tasks": artifact["tasks"][:1]}, [tasks[1]])

    duplicate_record: TaskImportanceRecord = artifact["tasks"][0]
    with pytest.raises(ContractValidationError, match="duplicate task_id=hotpot_ms_1"):
        _ = select_importance_records(
            {**artifact, "tasks": [duplicate_record, duplicate_record]},
            [tasks[0]],
        )

    stale_task = deepcopy(tasks[0])
    stale_task["memory_items"][0]["text"] = "Changed content."
    with pytest.raises(ContractValidationError, match="content_digest"):
        _ = select_importance_records(artifact, [stale_task])
