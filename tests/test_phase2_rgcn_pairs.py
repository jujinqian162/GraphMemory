from __future__ import annotations

import pytest

from graph_memory.datasets.selection import text_ranking_requests_for_dataset
from graph_memory.datasets.hotpotqa.records import (
    HotpotQALabelRecord,
    HotpotQARankingRecord,
)
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.graphs.contracts import EvidenceGraph, GraphItemNode
from graph_memory.training_pairs import build_train_pairs
from graph_memory.training_pairs.config import NegativeSamplingConfig
from graph_memory.training_pairs.contracts import TrainPairDataset
from graph_memory.training_pairs.requests import TrainPairBuildTask


def tiny_task_inputs() -> list[HotpotQARankingRecord]:
    return [
        HotpotQARankingRecord.model_validate(
            {
                "task_id": "hotpot_pair_test",
                "question": "Which city links the bridge evidence?",
                "candidate_sentences": [
                    {
                        "sentence_id": f"m{index}",
                        "text": text,
                        "title": title,
                        "sentence_index": 0,
                        "position": index,
                    }
                    for index, (title, text) in enumerate(
                        (
                            ("Alpha", "Alpha city is connected to the bridge."),
                            ("Beta", "The bridge evidence is in Beta."),
                            ("Gamma", "Gamma is a nearby distractor."),
                            ("Delta", "The answer depends on Delta."),
                        )
                    )
                ],
            }
        )
    ]


def tiny_labels() -> list[HotpotQALabelRecord]:
    return [
        HotpotQALabelRecord(
            task_id="hotpot_pair_test",
            gold_answer="Beta and Delta",
            gold_evidence_sentence_ids=("m1", "m3"),
            gold_dependency_edges=(),
        )
    ]


def tiny_graphs() -> list[EvidenceGraph]:
    task = tiny_task_inputs()[0]
    nodes = [
        GraphItemNode(
            id=sentence.sentence_id,
            node_kind="document_sentence",
            text=sentence.text,
            source_ref=sentence.title,
            group_key=f"document:{sentence.title}",
            sequence_index=sentence.sentence_index,
            metadata={
                "title": sentence.title,
                "position": sentence.position,
            },
        )
        for sentence in task.candidate_sentences
    ]
    return [
        EvidenceGraph.model_validate(
            {
                "task_id": task.task_id,
                "nodes": [
                    {"id": "q", "node_type": "question", "text": task.question},
                    *nodes,
                ],
                "edges": [
                    {
                        "source": "m1",
                        "target": "m2",
                        "edge_type": "bridge",
                        "weight": 1.0,
                        "directed": False,
                    },
                    {
                        "source": "m3",
                        "target": "m0",
                        "edge_type": "entity_overlap",
                        "weight": 1.0,
                        "directed": False,
                    },
                ],
            }
        )
    ]


def _pair_tasks() -> list[TrainPairBuildTask]:
    request = text_ranking_requests_for_dataset("hotpotqa", [tiny_task_inputs()[0]])[0]
    raw_label = tiny_labels()[0]
    label = EvidenceLabel(
        task_id=raw_label.task_id,
        gold_answer=raw_label.gold_answer,
        gold_evidence_item_ids=raw_label.gold_evidence_sentence_ids,
        gold_dependency_edges=(),
    )
    return [
        TrainPairBuildTask(text_request=request, label=label, graph=tiny_graphs()[0])
    ]


def test_train_pair_validation_rejects_question_node_sample() -> None:
    task = _pair_tasks()[0]
    assert task.graph is not None
    pairs = [
        {
            "task_id": task.text_request.task_id,
            "node_id": "m1",
            "label": 1,
            "sample_type": "positive",
        },
        {
            "task_id": task.text_request.task_id,
            "node_id": "q",
            "label": 0,
            "sample_type": "easy_random",
        },
    ]
    with pytest.raises(ValueError, match="query node"):
        TrainPairDataset.model_validate(
            {
                "pairs": pairs,
                "requests": (task.text_request,),
                "labels": (task.label,),
                "graphs": (task.graph,),
            }
        )


def test_build_train_pairs_creates_valid_positive_random_and_graph_neighbor_samples() -> (
    None
):
    result = build_train_pairs(
        _pair_tasks(),
        NegativeSamplingConfig(
            random_seed=7,
            easy_random_per_positive=1,
            hard_bm25_per_positive=0,
            hard_dense_per_positive=0,
            hard_graph_neighbor_per_positive=1,
            hard_pool_size=10,
        ),
    )
    assert result.summary.positive_count == 2
    assert result.summary.negative_count_by_type == {
        "easy_random": 2,
        "hard_graph_neighbor": 2,
    }
    assert all(pair.node_id != "q" for pair in result.pairs)
