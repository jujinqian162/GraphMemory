from __future__ import annotations

import json

import pytest

from graph_memory.datasets.longmemeval import convert_longmemeval_example, parse_longmemeval_example


def _raw_example() -> dict[str, object]:
    return {
        "question_id": "example_001",
        "question_type": "single-session-user",
        "question": "Where did I say I planned to meet Alex?",
        "answer": "At the library.",
        "question_date": "2024-01-10T12:00:00",
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": ["2024-01-01T09:00:00", "2024-01-03T15:00:00"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "Let's meet Alex at the library tomorrow.", "has_answer": True},
                {"role": "assistant", "content": "That sounds good."},
            ],
            [
                {"role": "user", "content": "A distractor memory."},
            ],
        ],
        "answer_session_ids": ["s1"],
    }


def test_convert_longmemeval_example_writes_turn_level_input_and_label_records() -> None:
    converted = convert_longmemeval_example(parse_longmemeval_example(_raw_example()))

    assert converted.ranking_record == {
        "task_id": "longmem_example_001",
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
            {
                "item_id": "m2",
                "session_id": "s2",
                "session_order": 1,
                "turn_index": 0,
                "global_position": 2,
                "role": "user",
                "datetime": "2024-01-03T15:00:00",
                "text": "A distractor memory.",
            },
        ],
        "metadata": {
            "dataset": "longmemeval_v1",
            "raw_id": "example_001",
            "question_type": "single-session-user",
            "candidate_granularity": "turn",
        },
    }
    assert converted.label_record == {
        "task_id": "longmem_example_001",
        "gold_answer": "At the library.",
        "gold_support_item_ids": ["m0"],
        "gold_support_session_ids": ["s1"],
        "gold_dependency_edges": [],
        "metadata": {
            "dataset": "longmemeval_v1",
            "raw_id": "example_001",
            "question_type": "single-session-user",
            "support_label_source": "has_answer",
        },
    }


def test_convert_longmemeval_ranking_record_contains_no_gold_fields() -> None:
    ranking_record = convert_longmemeval_example(parse_longmemeval_example(_raw_example())).ranking_record

    rendered = json.dumps(ranking_record, ensure_ascii=False, sort_keys=True)

    forbidden = [
        "answer",
        "has_answer",
        "answer_session_ids",
        "gold_support_item_ids",
        "gold_support_session_ids",
        "gold_dependency_edges",
        "gold",
    ]
    assert all(field not in rendered for field in forbidden)


def test_convert_longmemeval_example_rejects_examples_without_turn_support_labels() -> None:
    raw = _raw_example()
    raw["haystack_sessions"] = [
        [{"role": "user", "content": "No precise support."}],
        [{"role": "assistant", "content": "Still no precise support."}],
    ]

    with pytest.raises(ValueError, match="turn support labels"):
        convert_longmemeval_example(parse_longmemeval_example(raw))
