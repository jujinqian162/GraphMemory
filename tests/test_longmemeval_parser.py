from __future__ import annotations

import pytest

from graph_memory.datasets.longmemeval import parse_longmemeval_example, parse_longmemeval_examples


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


def test_parse_longmemeval_example_preserves_parallel_session_fields() -> None:
    example = parse_longmemeval_example(_raw_example())

    assert example.raw_id == "example_001"
    assert example.question_type == "single-session-user"
    assert example.question_datetime == "2024-01-10T12:00:00"
    assert example.sessions[0].session_id == "s1"
    assert example.sessions[0].datetime == "2024-01-01T09:00:00"
    assert example.sessions[0].turns[0].role == "user"
    assert example.sessions[0].turns[0].content == "Let's meet Alex at the library tomorrow."
    assert example.sessions[0].turns[0].has_answer is True
    assert example.sessions[0].turns[1].has_answer is False
    assert example.answer_session_ids == ("s1",)


def test_parse_longmemeval_examples_reports_record_index() -> None:
    raw_records = [_raw_example(), {**_raw_example(), "question_id": ""}]

    with pytest.raises(ValueError, match="index=1"):
        parse_longmemeval_examples(raw_records)


@pytest.mark.parametrize(
    "field_name",
    [
        "question_id",
        "question_type",
        "question",
        "answer",
        "question_date",
        "haystack_session_ids",
        "haystack_dates",
        "haystack_sessions",
        "answer_session_ids",
    ],
)
def test_parse_longmemeval_example_fails_fast_when_required_field_is_missing(field_name: str) -> None:
    raw = _raw_example()
    raw.pop(field_name)

    with pytest.raises(ValueError, match=field_name):
        parse_longmemeval_example(raw)


def test_parse_longmemeval_example_rejects_parallel_array_length_mismatch() -> None:
    raw = _raw_example()
    raw["haystack_dates"] = ["2024-01-01T09:00:00"]

    with pytest.raises(ValueError, match="parallel haystack arrays"):
        parse_longmemeval_example(raw)


def test_parse_longmemeval_example_rejects_turn_without_content() -> None:
    raw = _raw_example()
    raw["haystack_sessions"] = [[{"role": "user"}], [{"role": "user", "content": "ok"}]]

    with pytest.raises(ValueError, match="content"):
        parse_longmemeval_example(raw)


def test_parse_longmemeval_example_rejects_unknown_raw_fields() -> None:
    raw = {**_raw_example(), "leaky_extra": "value"}

    with pytest.raises(ValueError, match="unknown fields"):
        parse_longmemeval_example(raw)
