from typing import TypeAlias

import pytest

from graph_memory.datasets.hotpotqa import HotpotQAConversionResult, convert_hotpotqa_examples, parse_hotpotqa_examples
from graph_memory.datasets.hotpotqa.compatibility import combined_hotpotqa_records
from graph_memory.datasets.hotpotqa.parser import parse_hotpotqa_example
from graph_memory.datasets.splits import sample_split

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
    assert inputs[0]["question"] == "Where is the Eiffel Tower and what river runs through that city?"
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


def test_convert_hotpotqa_requires_raw_id():
    raw = hotpot_raw_example()
    del raw["_id"]

    with pytest.raises(ValueError, match="_id"):
        parse_hotpotqa_examples([raw])


def test_parse_hotpotqa_rejects_non_text_sentence():
    malformed = {**hotpot_raw_example(), "context": [["Ada Lovelace", ["ok", 3]]], "_id": "abc123"}

    with pytest.raises(ValueError, match="must be text"):
        parse_hotpotqa_example(malformed)


def test_combined_hotpotqa_records_joins_ranking_and_label_by_task():
    parsed = parse_hotpotqa_examples([hotpot_raw_example()])
    conversion = convert_hotpotqa_examples(parsed)
    ranking_record = conversion.ranking_records[0]
    label_record = conversion.label_records[0]

    assert combined_hotpotqa_records([ranking_record], [label_record]) == [
        {**ranking_record, **label_record}
    ]


def test_convert_hotpotqa_fails_when_supporting_fact_cannot_map():
    raw = hotpot_raw_example()
    raw["supporting_facts"] = [["Missing Title", 0]]

    parsed_examples = parse_hotpotqa_examples([raw])
    with pytest.raises(ValueError, match="supporting fact"):
        convert_hotpotqa_examples(parsed_examples)


def test_sample_split_is_deterministic_for_same_seed_and_offset():
    examples = [{"_id": str(index)} for index in range(10)]

    assert sample_split(examples, count=4, seed=13, offset=2) == sample_split(
        examples, count=4, seed=13, offset=2
    )


def test_sample_split_uses_offset_for_disjoint_slices():
    examples = [{"_id": str(index)} for index in range(20)]

    dev = sample_split(examples, count=5, seed=13, offset=0)
    test = sample_split(examples, count=5, seed=13, offset=5)

    assert {example["_id"] for example in dev}.isdisjoint({example["_id"] for example in test})


@pytest.mark.parametrize(
    ("count", "offset"),
    [(-1, 0), (1, -1), (5, 6)],
)
def test_sample_split_rejects_invalid_bounds(count: int, offset: int):
    examples = [{"_id": str(index)} for index in range(10)]

    with pytest.raises(ValueError):
        sample_split(examples, count=count, seed=13, offset=offset)
