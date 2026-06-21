import math
from typing import cast

import pytest

from graph_memory.contracts.graphs import GraphItemNode
from graph_memory.datasets.hotpotqa.projectors import HotpotQAToTextRankingRequest
from graph_memory.datasets.hotpotqa.records import HotpotQARankingRecord
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.validation import (
    ContractValidationError,
    validate_graphs,
    validate_hotpotqa_label_records,
    validate_hotpotqa_ranking_records,
    validate_ranked_results,
)


def valid_task_inputs() -> list[HotpotQARankingRecord]:
    return [
        {
            "task_id": "hotpot_ex1",
            "question": "Where is the Eiffel Tower?",
            "candidate_sentences": [
                {
                    "sentence_id": "m0",
                    "title": "Eiffel Tower",
                    "sentence_index": 0,
                    "position": 0,
                    "text": "The Eiffel Tower is in Paris.",
                },
                {
                    "sentence_id": "m1",
                    "title": "Paris",
                    "sentence_index": 0,
                    "position": 1,
                    "text": "Paris is in France.",
                },
            ],
        }
    ]


def _graph_nodes(record: HotpotQARankingRecord) -> list[GraphItemNode]:
    return [
        {
            "id": sentence["sentence_id"],
            "node_type": "graph_item",
            "node_kind": "document_sentence",
            "text": sentence["text"],
            "source_ref": sentence["title"],
            "group_key": f"document:{sentence['title']}",
            "sequence_index": sentence["sentence_index"],
            "metadata": {"title": sentence["title"], "position": sentence["position"]},
        }
        for sentence in record["candidate_sentences"]
    ]


def inputs_by_task_id() -> dict[str, HotpotQARankingRecord]:
    return {task_input["task_id"]: task_input for task_input in valid_task_inputs()}


def ranking_requests() -> list[TextRankingRequest]:
    projector = HotpotQAToTextRankingRequest()
    return [projector.project(task_input) for task_input in valid_task_inputs()]

def test_input_validation_rejects_label_leakage():
    task_inputs = valid_task_inputs()
    task_input = cast(dict[str, object], cast(object, task_inputs[0]))
    task_input["gold_evidence_sentence_ids"] = ["m0"]

    with pytest.raises(ContractValidationError, match="gold_evidence_sentence_ids"):
        validate_hotpotqa_ranking_records(task_inputs)


def test_label_validation_rejects_task_id_mismatch():
    labels = [
        {
            "task_id": "hotpot_missing",
            "gold_answer": "Paris",
            "gold_evidence_sentence_ids": ["m0"],
            "gold_dependency_edges": [],
        }
    ]

    with pytest.raises(ContractValidationError, match="hotpot_missing"):
        validate_hotpotqa_label_records(labels, inputs_by_task_id())


def test_graph_validation_rejects_missing_edge_endpoint():
    graphs = [
        {
            "task_id": "hotpot_ex1",
            "nodes": [
                {"id": "q", "node_type": "question", "text": "Where is the Eiffel Tower?"},
                *_graph_nodes(valid_task_inputs()[0]),
            ],
            "edges": [
                {
                    "source": "m0",
                    "target": "m9",
                    "edge_type": "bridge",
                    "weight": 1.0,
                    "directed": False,
                }
            ],
        }
    ]

    with pytest.raises(ContractValidationError, match="m9"):
        validate_graphs(graphs, ranking_requests())


def test_ranked_result_validation_rejects_duplicate_ranked_node():
    predictions = [
        {
            "task_id": "hotpot_ex1",
            "method": "bm25",
            "ranked_nodes": [
                {"node_id": "m0", "score": 2.0},
                {"node_id": "m0", "score": 1.0},
            ],
            "retrieved_subgraph": {"nodes": ["m0"], "edges": []},
            "latency_ms": 1.0,
            "input_tokens": 0,
        }
    ]

    with pytest.raises(ContractValidationError, match="duplicate"):
        validate_ranked_results(predictions, ranking_requests())


def test_ranked_result_validation_rejects_path_metric_capability_metadata():
    predictions = [
        {
            "task_id": "hotpot_ex1",
            "method": "bm25",
            "ranked_nodes": [{"node_id": "m0", "score": 2.0}, {"node_id": "m1", "score": 1.0}],
            "retrieved_subgraph": {"nodes": ["m0", "m1"], "edges": []},
            "latency_ms": 1.0,
            "input_tokens": 0,
            "metadata": {"path_metrics_supported": True},
        }
    ]

    with pytest.raises(ContractValidationError, match="method registry"):
        validate_ranked_results(predictions, ranking_requests())


def test_ranked_result_validation_rejects_non_finite_scores():
    predictions = [
        {
            "task_id": "hotpot_ex1",
            "method": "bm25",
            "ranked_nodes": [
                {"node_id": "m0", "score": 2.0},
                {"node_id": "m1", "score": math.inf},
            ],
            "retrieved_subgraph": {"nodes": ["m0", "m1"], "edges": []},
            "latency_ms": 1.0,
            "input_tokens": 0,
        }
    ]

    with pytest.raises(ContractValidationError, match="finite"):
        validate_ranked_results(predictions, ranking_requests())
