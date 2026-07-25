import json
from pathlib import Path
from typing import TypeAlias

import pytest

from graph_memory.datasets.hotpotqa import (
    HotpotQAConversionResult,
    convert_hotpotqa_examples,
    parse_hotpotqa_examples,
)
from graph_memory.datasets.hotpotqa.parser import parse_hotpotqa_example
from graph_memory.datasets.splits import sample_split
from graph_memory.stages.prepare import prepare_split

RawHotpotQARecord: TypeAlias = dict[str, object]


def hotpot_raw_example() -> RawHotpotQARecord:
    return {
        "_id": "ex1",
        "question": "Where is the Eiffel Tower and what river runs through that city?",
        "answer": "Paris and the Seine",
        "context": [
            ["Eiffel Tower", ["The Eiffel Tower is in Paris.", "It opened in 1889."]],
            ["Paris", ["Paris is in France.", "The Seine runs through Paris."]],
        ],
        "supporting_facts": [["Eiffel Tower", 0], ["Paris", 1]],
    }


def test_supporting_facts_map_title_sentence_to_node_ids():
    parsed_examples = parse_hotpotqa_examples([hotpot_raw_example()])
    conversion = convert_hotpotqa_examples(parsed_examples)

    assert isinstance(conversion, HotpotQAConversionResult)
    inputs = conversion.ranking_records
    labels = conversion.label_records
    assert inputs[0]["task_id"] == "hotpot_ex1"
    assert (
        inputs[0]["question"]
        == "Where is the Eiffel Tower and what river runs through that city?"
    )
    assert inputs[0]["candidate_sentences"][0]["sentence_id"] == "m0"
    assert inputs[0]["candidate_sentences"][0]["sentence_index"] == 0
    assert inputs[0]["candidate_sentences"][0]["position"] == 0
    assert inputs[0]["candidate_sentences"][3]["sentence_id"] == "m3"
    assert inputs[0]["candidate_sentences"][3]["sentence_index"] == 1
    assert inputs[0]["candidate_sentences"][3]["position"] == 3
    assert labels[0]["task_id"] == "hotpot_ex1"
    assert labels[0]["gold_answer"] == "Paris and the Seine"
    assert labels[0]["gold_evidence_sentence_ids"] == ["m0", "m3"]
    assert labels[0]["gold_dependency_edges"] == []
    assert "gold_answer" not in inputs[0]
    assert "gold_evidence_sentence_ids" not in inputs[0]
    assert "supporting_facts" not in inputs[0]


def test_hotpotqa_parse_and_convert_reject_invalid_records() -> None:
    missing_id = hotpot_raw_example()
    del missing_id["_id"]
    with pytest.raises(ValueError, match="_id"):
        parse_hotpotqa_examples([missing_id])

    non_text = {
        **hotpot_raw_example(),
        "context": [["Ada Lovelace", ["ok", 3]]],
        "_id": "abc123",
    }
    with pytest.raises(ValueError, match="must be text"):
        parse_hotpotqa_example(non_text)

    empty_sentence = {
        **hotpot_raw_example(),
        "context": [["Ada Lovelace", [""]]],
        "_id": "empty_sentence",
    }
    with pytest.raises(ValueError, match="sentence_id=0 must be a non-empty string"):
        _ = parse_hotpotqa_example(empty_sentence)

    unmapped = hotpot_raw_example()
    unmapped["supporting_facts"] = [["Missing Title", 0]]
    with pytest.raises(ValueError, match="supporting fact"):
        convert_hotpotqa_examples(parse_hotpotqa_examples([unmapped]))


def test_prepare_hotpotqa_drops_record_with_empty_candidate_sentence(tmp_path: Path) -> None:
    invalid = {
        **hotpot_raw_example(),
        "_id": "empty_sentence",
        "context": [["Ada Lovelace", [""]]],
        "supporting_facts": [["Ada Lovelace", 0]],
    }
    source = tmp_path / "hotpotqa.json"
    _ = source.write_text(json.dumps([hotpot_raw_example(), invalid]), encoding="utf-8")

    prepared = prepare_split(
        "hotpotqa",
        source,
        count=None,
        seed=13,
        offset=0,
        strict_invalid_examples=False,
    )

    assert prepared.counts["raw_examples"] == 2
    assert prepared.counts["valid_examples"] == 1
    assert prepared.counts["invalid_examples_dropped"] == 1
    assert prepared.counts["task_inputs"] == 1
    assert isinstance(prepared.task_inputs[0], dict)
    assert prepared.task_inputs[0]["task_id"] == "hotpot_ex1"


def test_sample_split_is_deterministic_disjoint_and_bounds_checked() -> None:
    examples = [{"_id": str(index)} for index in range(20)]

    first = sample_split(examples, count=5, seed=13, offset=0)
    second = sample_split(examples, count=5, seed=13, offset=0)
    assert first == second

    later = sample_split(examples, count=5, seed=13, offset=5)
    assert {example["_id"] for example in first}.isdisjoint(
        {example["_id"] for example in later}
    )

    for count, offset in ((-1, 0), (1, -1), (5, 6)):
        with pytest.raises(ValueError):
            sample_split(
                [{"_id": str(index)} for index in range(10)],
                count=count,
                seed=13,
                offset=offset,
            )
