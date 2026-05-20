import pytest

from graph_memory.hotpotqa import convert_hotpotqa_examples
from graph_memory.splits import sample_split


def hotpot_raw_example() -> dict:
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
    inputs, labels = convert_hotpotqa_examples([hotpot_raw_example()])

    assert inputs[0]["task_id"] == "hotpot_ex1"
    assert inputs[0]["query"] == "Where is the Eiffel Tower and what river runs through that city?"
    assert inputs[0]["memory_items"][0]["id"] == "m0"
    assert inputs[0]["memory_items"][0]["sentence_id"] == 0
    assert inputs[0]["memory_items"][0]["position"] == 0
    assert inputs[0]["memory_items"][3]["id"] == "m3"
    assert inputs[0]["memory_items"][3]["sentence_id"] == 1
    assert inputs[0]["memory_items"][3]["position"] == 3
    assert labels[0]["task_id"] == "hotpot_ex1"
    assert labels[0]["gold_answer"] == "Paris and the Seine"
    assert labels[0]["gold_evidence_nodes"] == ["m0", "m3"]
    assert labels[0]["gold_dependency_edges"] == []
    assert "gold_answer" not in inputs[0]
    assert "gold_evidence_nodes" not in inputs[0]
    assert "supporting_facts" not in inputs[0]


def test_convert_hotpotqa_requires_raw_id():
    raw = hotpot_raw_example()
    del raw["_id"]

    with pytest.raises(ValueError, match="_id"):
        convert_hotpotqa_examples([raw])


def test_convert_hotpotqa_fails_when_supporting_fact_cannot_map():
    raw = hotpot_raw_example()
    raw["supporting_facts"] = [["Missing Title", 0]]

    with pytest.raises(ValueError, match="supporting fact"):
        convert_hotpotqa_examples([raw])


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
