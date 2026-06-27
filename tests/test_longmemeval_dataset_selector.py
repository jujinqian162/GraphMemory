from __future__ import annotations

import pytest
from typing import cast

from graph_memory.contracts.errors import ContractValidationError
from graph_memory.datasets.selection import (
    evidence_labels_for_dataset,
    graph_build_requests_for_dataset,
    temporal_memory_requests_for_dataset,
    text_ranking_requests_for_dataset,
    validate_label_records_for_dataset,
    validate_ranking_records_for_dataset,
)


def _ranking_record() -> dict[str, object]:
    return {
        "task_id": "longmem_q1",
        "question": "Where did I say I planned to meet Alex?",
        "question_datetime": "2024-01-10T12:00:00",
        "candidate_items": [
            {
                "item_id": "m0",
                "session_id": "s1",
                "session_order": 0,
                "turn_index": 0,
                "global_position": 0,
                "role": "user",
                "datetime": "2024-01-01T09:00:00",
                "text": "Let's meet Alex at the library tomorrow.",
            },
            {
                "item_id": "m1",
                "session_id": "s1",
                "session_order": 0,
                "turn_index": 1,
                "global_position": 1,
                "role": "assistant",
                "datetime": "2024-01-01T09:00:00",
                "text": "That sounds good.",
            },
        ],
        "metadata": {
            "dataset": "longmemeval_v1",
            "raw_id": "q1",
            "question_type": "single-session-user",
            "candidate_granularity": "turn",
        },
    }


def _label_record() -> dict[str, object]:
    return {
        "task_id": "longmem_q1",
        "gold_answer": "At the library.",
        "gold_support_item_ids": ["m0"],
        "gold_support_session_ids": ["s1"],
        "gold_dependency_edges": [],
        "metadata": {
            "dataset": "longmemeval_v1",
            "raw_id": "q1",
            "question_type": "single-session-user",
            "support_label_source": "has_answer",
        },
    }


def test_longmemeval_dataset_selector_validates_and_projects_requests() -> None:
    ranking_records = [_ranking_record()]
    label_records = [_label_record()]
    records_by_task_id = {"longmem_q1": ranking_records[0]}

    validate_ranking_records_for_dataset("longmemeval", ranking_records)
    validate_label_records_for_dataset("longmemeval", label_records, records_by_task_id)

    assert text_ranking_requests_for_dataset("longmemeval", ranking_records)[0].task_id == "longmem_q1"
    assert temporal_memory_requests_for_dataset("longmemeval", ranking_records)[0].importance_by_item_id == {
        "m0": 0.0,
        "m1": 0.0,
    }
    assert graph_build_requests_for_dataset("longmemeval", ranking_records)[0].nodes[0].group_key == "session:s1"
    assert evidence_labels_for_dataset("longmemeval", label_records)[0].gold_evidence_item_ids == ("m0",)


def test_longmemeval_ranking_validation_rejects_label_leakage() -> None:
    ranking_record = _ranking_record()
    leaky_record = {
        **ranking_record,
        "candidate_items": [
            {
                **cast(list[dict[str, object]], ranking_record["candidate_items"])[0],
                "has_answer": True,
            }
        ],
    }

    with pytest.raises(ContractValidationError, match="has_answer"):
        validate_ranking_records_for_dataset("longmemeval", [leaky_record])


def test_longmemeval_label_validation_rejects_unknown_support_items() -> None:
    label = {**_label_record(), "gold_support_item_ids": ["missing"]}

    with pytest.raises(ContractValidationError, match="does not exist"):
        validate_label_records_for_dataset("longmemeval", [label], {"longmem_q1": _ranking_record()})
