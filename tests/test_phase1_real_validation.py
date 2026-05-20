import math

import pytest

from graph_memory.validation import (
    ContractValidationError,
    validate_graphs,
    validate_memory_task_inputs,
    validate_memory_task_labels,
    validate_ranked_results,
)


def valid_task_inputs() -> list[dict]:
    return [
        {
            "task_id": "hotpot_ex1",
            "query": "Where is the Eiffel Tower?",
            "memory_items": [
                {
                    "id": "m0",
                    "node_type": "document_sentence",
                    "text": "The Eiffel Tower is in Paris.",
                    "source": "Eiffel Tower",
                    "sentence_id": 0,
                    "position": 0,
                },
                {
                    "id": "m1",
                    "node_type": "document_sentence",
                    "text": "Paris is in France.",
                    "source": "Paris",
                    "sentence_id": 0,
                    "position": 1,
                },
            ],
        }
    ]


def inputs_by_task_id() -> dict[str, dict]:
    return {task_input["task_id"]: task_input for task_input in valid_task_inputs()}


def test_input_validation_rejects_label_leakage():
    task_inputs = valid_task_inputs()
    task_inputs[0]["gold_evidence_nodes"] = ["m0"]

    with pytest.raises(ContractValidationError, match="gold_evidence_nodes"):
        validate_memory_task_inputs(task_inputs)


def test_label_validation_rejects_task_id_mismatch():
    labels = [
        {
            "task_id": "hotpot_missing",
            "gold_answer": "Paris",
            "gold_evidence_nodes": ["m0"],
            "gold_dependency_edges": [],
        }
    ]

    with pytest.raises(ContractValidationError, match="hotpot_missing"):
        validate_memory_task_labels(labels, inputs_by_task_id())


def test_graph_validation_rejects_missing_edge_endpoint():
    graphs = [
        {
            "task_id": "hotpot_ex1",
            "nodes": [
                {"id": "q", "node_type": "question", "text": "Where is the Eiffel Tower?"},
                *valid_task_inputs()[0]["memory_items"],
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
        validate_graphs(graphs, inputs_by_task_id())


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
        validate_ranked_results(predictions, inputs_by_task_id())


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
        validate_ranked_results(predictions, inputs_by_task_id())
