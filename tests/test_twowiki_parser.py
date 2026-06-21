from __future__ import annotations

import pytest

from graph_memory.datasets.twowiki import parse_twowiki_example, parse_twowiki_examples


def _raw_example() -> dict[str, object]:
    return {
        "_id": "abc123",
        "type": "compositional",
        "question": "Who is Ada's mother?",
        "context": [
            ["Film A", ["Film A was directed by Ada.", "A distractor sentence."]],
            ["Ada Lovelace", ["Ada was the daughter of Beth."]],
        ],
        "supporting_facts": [["Film A", 0], ["Ada Lovelace", 0]],
        "evidences": [["Film A", "director", "Ada"], ["Ada", "mother", "Beth"]],
        "evidences_id": [["film-a", "director", "ada"], ["ada", "mother", "beth"]],
        "answer_id": "beth",
        "answer": "Beth",
    }


def test_parse_twowiki_example_preserves_path_label_fields() -> None:
    example = parse_twowiki_example(_raw_example())

    assert example.raw_id == "abc123"
    assert example.question_type == "compositional"
    assert example.documents[0].title == "Film A"
    assert example.supporting_facts[1].title == "Ada Lovelace"
    assert example.evidences[0].subject == "Film A"
    assert example.evidences[1].object == "Beth"
    assert example.evidences_id[0].subject == "film-a"
    assert example.answer_id == "beth"


def test_parse_twowiki_examples_reports_record_index() -> None:
    raw_records = [_raw_example(), {**_raw_example(), "_id": ""}]

    with pytest.raises(ValueError, match="index=1"):
        parse_twowiki_examples(raw_records)


@pytest.mark.parametrize("field_name", ["_id", "question", "context", "supporting_facts", "evidences", "type"])
def test_parse_twowiki_example_fails_fast_when_required_field_is_missing(field_name: str) -> None:
    raw = _raw_example()
    raw.pop(field_name)

    with pytest.raises(ValueError, match=field_name):
        parse_twowiki_example(raw)


def test_parse_twowiki_example_rejects_empty_document() -> None:
    raw = _raw_example()
    raw["context"] = [["Empty", []]]

    with pytest.raises(ValueError, match="non-empty"):
        parse_twowiki_example(raw)


def test_parse_twowiki_example_rejects_non_string_sentence() -> None:
    raw = _raw_example()
    raw["context"] = [["Bad", ["valid", 13]]]

    with pytest.raises(ValueError, match="sentence_id=1"):
        parse_twowiki_example(raw)


def test_parse_twowiki_example_rejects_unknown_raw_fields() -> None:
    raw = {**_raw_example(), "leaky_extra": "value"}

    with pytest.raises(ValueError, match="unknown fields"):
        parse_twowiki_example(raw)
