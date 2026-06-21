from __future__ import annotations

import json

from graph_memory.datasets.twowiki import convert_twowiki_example, parse_twowiki_example


def _raw_compositional_example() -> dict[str, object]:
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
        "answer": "Beth",
    }


def _raw_comparison_example() -> dict[str, object]:
    return {
        "_id": "cmp1",
        "type": "comparison",
        "question": "Which film was released earlier?",
        "context": [
            ["Film A", ["Film A was released in 2003."]],
            ["Film B", ["Film B was released in 1932."]],
        ],
        "supporting_facts": [["Film A", 0], ["Film B", 0]],
        "evidences": [["Film A", "publication date", "2003"], ["Film B", "publication date", "1932"]],
        "answer": "Film B",
    }


def test_convert_twowiki_example_writes_separated_input_and_label_records() -> None:
    converted = convert_twowiki_example(parse_twowiki_example(_raw_compositional_example()))

    assert converted.ranking_record == {
        "task_id": "2wiki_abc123",
        "question": "Who is Ada's mother?",
        "question_type": "compositional",
        "candidate_sentences": [
            {
                "sentence_id": "m0",
                "title": "Film A",
                "sentence_index": 0,
                "position": 0,
                "text": "Film A was directed by Ada.",
            },
            {
                "sentence_id": "m1",
                "title": "Film A",
                "sentence_index": 1,
                "position": 1,
                "text": "A distractor sentence.",
            },
            {
                "sentence_id": "m2",
                "title": "Ada Lovelace",
                "sentence_index": 0,
                "position": 2,
                "text": "Ada was the daughter of Beth.",
            },
        ],
        "metadata": {"dataset": "2wiki", "raw_id": "abc123"},
    }
    assert converted.label_record == {
        "task_id": "2wiki_abc123",
        "gold_answer": "Beth",
        "gold_evidence_sentence_ids": ["m0", "m2"],
        "gold_dependency_edges": [["m0", "m2"]],
        "metadata": {
            "question_type": "compositional",
            "path_label_source": "evidences",
            "path_supported": True,
            "mapping_ambiguity_count": 0,
        },
    }


def test_convert_twowiki_example_keeps_label_only_fields_out_of_ranking_record() -> None:
    converted = convert_twowiki_example(parse_twowiki_example(_raw_compositional_example()))
    serialized_ranking_record = json.dumps(converted.ranking_record, sort_keys=True)

    forbidden = [
        "gold_answer",
        "supporting_facts",
        "evidences",
        "evidences_id",
        "answer_id",
        "gold_dependency_edges",
        "is_gold",
    ]
    assert all(field not in serialized_ranking_record for field in forbidden)


def test_convert_twowiki_comparison_example_does_not_force_dependency_edges() -> None:
    converted = convert_twowiki_example(parse_twowiki_example(_raw_comparison_example()))

    assert converted.label_record["gold_evidence_sentence_ids"] == ["m0", "m1"]
    assert converted.label_record["gold_dependency_edges"] == []
    assert converted.label_record["metadata"]["path_supported"] is False


def test_convert_twowiki_example_prefers_evidences_id_for_dependency_edges() -> None:
    raw = {
        **_raw_compositional_example(),
        "evidences": [["unmatched-title", "director", "unmatched-person"], ["unmatched-person", "mother", "Beth"]],
        "evidences_id": [["Film A", "director", "Ada"], ["Ada", "mother", "Beth"]],
    }

    converted = convert_twowiki_example(parse_twowiki_example(raw))

    assert converted.label_record["gold_dependency_edges"] == [["m0", "m2"]]
    assert converted.label_record["metadata"]["path_label_source"] == "evidences_id"
    assert converted.label_record["metadata"]["path_supported"] is True
