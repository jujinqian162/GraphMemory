from __future__ import annotations

import pytest

from graph_memory.datasets.musique import parse_musique_example, parse_musique_examples


def _raw_example() -> dict[str, object]:
    return {
        "id": "2hop__1_2",
        "question": "Where was the director of Film A born?",
        "answer": "London",
        "answer_aliases": ["London, England"],
        "answerable": True,
        "paragraphs": [
            {
                "idx": 0,
                "title": "Film A",
                "paragraph_text": "Film A was directed by Ada.",
                "is_supporting": True,
            },
            {
                "idx": 1,
                "title": "Ada Lovelace",
                "paragraph_text": "Ada Lovelace was born in London.",
                "is_supporting": True,
            },
            {
                "idx": 2,
                "title": "Distractor",
                "paragraph_text": "This paragraph is not useful.",
                "is_supporting": False,
            },
        ],
        "question_decomposition": [
            {
                "id": 1,
                "question": "Who directed Film A?",
                "answer": "Ada Lovelace",
                "paragraph_support_idx": 0,
            },
            {
                "id": 2,
                "question": "Where was #1 born?",
                "answer": "London",
                "paragraph_support_idx": 1,
            },
        ],
    }


def test_parse_musique_example_preserves_official_fields() -> None:
    example = parse_musique_example(_raw_example())

    assert example.raw_id == "2hop__1_2"
    assert example.question == "Where was the director of Film A born?"
    assert example.answer_aliases == ("London, England",)
    assert example.answerable is True
    assert example.paragraphs[0].idx == 0
    assert example.paragraphs[1].title == "Ada Lovelace"
    assert example.question_decomposition[0].step_id == "1"
    assert example.question_decomposition[1].question == "Where was #1 born?"
    assert example.question_decomposition[1].paragraph_support_idx == 1


def test_parse_musique_examples_reports_record_index() -> None:
    raw_records = [_raw_example(), {**_raw_example(), "id": ""}]

    with pytest.raises(ValueError, match="index=1"):
        parse_musique_examples(raw_records)


@pytest.mark.parametrize("field_name", ["id", "question", "answer", "answer_aliases", "answerable", "paragraphs", "question_decomposition"])
def test_parse_musique_example_fails_fast_when_required_field_is_missing(field_name: str) -> None:
    raw = _raw_example()
    raw.pop(field_name)

    with pytest.raises(ValueError, match=field_name):
        parse_musique_example(raw)


def test_parse_musique_example_rejects_full_unanswerable_records() -> None:
    raw = {**_raw_example(), "answerable": False}

    with pytest.raises(ValueError, match="answerable"):
        parse_musique_example(raw)


def test_parse_musique_example_rejects_duplicate_paragraph_idx() -> None:
    raw = _raw_example()
    paragraphs = list(raw["paragraphs"])  # type: ignore[arg-type]
    paragraphs[1] = {**paragraphs[1], "idx": 0}  # type: ignore[index]
    raw["paragraphs"] = paragraphs

    with pytest.raises(ValueError, match="duplicate paragraph idx"):
        parse_musique_example(raw)


def test_parse_musique_example_rejects_unknown_raw_fields() -> None:
    raw = {**_raw_example(), "leaky_extra": "value"}

    with pytest.raises(ValueError, match="unknown fields"):
        parse_musique_example(raw)
