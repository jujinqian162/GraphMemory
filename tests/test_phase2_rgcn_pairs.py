import json

import pytest

import graph_memory.learned.data as pair_data
from graph_memory.learned.data import build_train_pairs
from graph_memory.types import MemoryGraph, MemoryTaskInput, MemoryTaskLabels, NegativeSamplingConfig
from graph_memory.validation import ContractValidationError, validate_train_pairs
from scripts.build_train_pairs import main as build_train_pairs_main


def tiny_task_inputs() -> list[MemoryTaskInput]:
    return [
        {
            "task_id": "hotpot_pair_test",
            "query": "Which city links the bridge evidence?",
            "memory_items": [
                {
                    "id": "m0",
                    "node_type": "document_sentence",
                    "text": "Alpha city is connected to the bridge.",
                    "source": "Alpha",
                    "sentence_id": 0,
                    "position": 0,
                },
                {
                    "id": "m1",
                    "node_type": "document_sentence",
                    "text": "The bridge evidence is in Beta.",
                    "source": "Beta",
                    "sentence_id": 0,
                    "position": 1,
                },
                {
                    "id": "m2",
                    "node_type": "document_sentence",
                    "text": "Gamma is a nearby distractor.",
                    "source": "Gamma",
                    "sentence_id": 0,
                    "position": 2,
                },
                {
                    "id": "m3",
                    "node_type": "document_sentence",
                    "text": "The answer depends on Delta.",
                    "source": "Delta",
                    "sentence_id": 0,
                    "position": 3,
                },
            ],
        }
    ]


def tiny_labels() -> list[MemoryTaskLabels]:
    return [
        {
            "task_id": "hotpot_pair_test",
            "gold_answer": "Beta and Delta",
            "gold_evidence_nodes": ["m1", "m3"],
            "gold_dependency_edges": [],
        }
    ]


def tiny_graphs() -> list[MemoryGraph]:
    task = tiny_task_inputs()[0]
    return [
        {
            "task_id": task["task_id"],
            "nodes": [
                {"id": "q", "node_type": "question", "text": task["query"]},
                *task["memory_items"],
            ],
            "edges": [
                {"source": "m1", "target": "m2", "edge_type": "bridge", "weight": 1.0, "directed": False},
                {"source": "m3", "target": "m0", "edge_type": "entity_overlap", "weight": 1.0, "directed": False},
            ],
        }
    ]


def by_task_id(records):
    return {record["task_id"]: record for record in records}


def test_train_pair_validation_rejects_question_node_sample():
    pairs = [
        {"task_id": "hotpot_pair_test", "node_id": "m1", "label": 1, "sample_type": "positive"},
        {"task_id": "hotpot_pair_test", "node_id": "q", "label": 0, "sample_type": "easy_random"},
    ]

    with pytest.raises(ContractValidationError, match="q"):
        validate_train_pairs(pairs, by_task_id(tiny_task_inputs()), by_task_id(tiny_labels()), by_task_id(tiny_graphs()))


def test_build_train_pairs_creates_valid_positive_random_and_graph_neighbor_samples():
    config = NegativeSamplingConfig(
        random_seed=7,
        easy_random_per_positive=1,
        hard_bm25_per_positive=0,
        hard_dense_per_positive=0,
        hard_graph_neighbor_per_positive=1,
        hard_pool_size=10,
    )

    result = build_train_pairs(tiny_task_inputs(), tiny_labels(), tiny_graphs(), config)

    assert result.summary["positive_count"] == 2
    assert result.summary["negative_count_by_type"] == {"easy_random": 2, "hard_graph_neighbor": 2}
    assert {pair["node_id"] for pair in result.pairs if pair["label"] == 1} == {"m1", "m3"}
    assert all(pair["node_id"] != "q" for pair in result.pairs)
    validate_train_pairs(
        result.pairs,
        by_task_id(tiny_task_inputs()),
        by_task_id(tiny_labels()),
        by_task_id(tiny_graphs()),
    )


def test_build_train_pairs_cli_writes_pairs_summary_and_run_summary(tmp_path):
    tasks_path = tmp_path / "train_memory_tasks.input.json"
    labels_path = tmp_path / "train_memory_tasks.labels.json"
    graphs_path = tmp_path / "train_graphs.json"
    output_path = tmp_path / "train_pairs.json"
    tasks_path.write_text(json.dumps(tiny_task_inputs()), encoding="utf-8")
    labels_path.write_text(json.dumps(tiny_labels()), encoding="utf-8")
    graphs_path.write_text(json.dumps(tiny_graphs()), encoding="utf-8")

    exit_code = build_train_pairs_main(
        [
            "--tasks",
            str(tasks_path),
            "--labels",
            str(labels_path),
            "--graphs",
            str(graphs_path),
            "--output",
            str(output_path),
            "--random_seed",
            "7",
            "--easy_random_per_positive",
            "1",
            "--hard_bm25_per_positive",
            "0",
            "--hard_dense_per_positive",
            "0",
            "--hard_graph_neighbor_per_positive",
            "1",
        ]
    )

    assert exit_code == 0
    pairs = json.loads(output_path.read_text(encoding="utf-8"))
    summary = json.loads(output_path.with_name("train_pairs.summary.json").read_text(encoding="utf-8"))
    run_summary = json.loads(output_path.with_name("train_pairs.run_summary.json").read_text(encoding="utf-8"))
    assert summary["positive_count"] == 2
    assert summary["negative_count_by_type"]["hard_graph_neighbor"] == 2
    assert run_summary["status"] == "success"
    assert all(pair["node_id"] != "q" for pair in pairs)


def test_build_train_pairs_cli_reads_pair_sampling_from_config(tmp_path):
    tasks_path = tmp_path / "train_memory_tasks.input.json"
    labels_path = tmp_path / "train_memory_tasks.labels.json"
    graphs_path = tmp_path / "train_graphs.json"
    output_path = tmp_path / "train_pairs.json"
    config_path = tmp_path / "effective_training_config.json"
    tasks_path.write_text(json.dumps(tiny_task_inputs()), encoding="utf-8")
    labels_path.write_text(json.dumps(tiny_labels()), encoding="utf-8")
    graphs_path.write_text(json.dumps(tiny_graphs()), encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "method": "dense_rgcn_graph_retriever",
                "profile": "quick",
                "pair_sampling": {
                    "random_seed": 7,
                    "easy_random_per_positive": 1,
                    "hard_bm25_per_positive": 0,
                    "hard_dense_per_positive": 0,
                    "hard_graph_neighbor_per_positive": 1,
                    "hard_pool_size": 10,
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = build_train_pairs_main(
        [
            "--tasks",
            str(tasks_path),
            "--labels",
            str(labels_path),
            "--graphs",
            str(graphs_path),
            "--output",
            str(output_path),
            "--config",
            str(config_path),
        ]
    )

    assert exit_code == 0
    summary = json.loads(output_path.with_name("train_pairs.summary.json").read_text(encoding="utf-8"))
    run_summary = json.loads(output_path.with_name("train_pairs.run_summary.json").read_text(encoding="utf-8"))
    assert summary["negative_count_by_type"] == {"easy_random": 2, "hard_graph_neighbor": 2}
    assert run_summary["effective_config"]["hard_dense_per_positive"] == 0
    assert run_summary["effective_config"]["hard_graph_neighbor_per_positive"] == 1


def test_build_train_pairs_cli_uses_config_encoder_for_hard_dense_negatives(tmp_path, monkeypatch):
    tasks_path = tmp_path / "train_memory_tasks.input.json"
    labels_path = tmp_path / "train_memory_tasks.labels.json"
    graphs_path = tmp_path / "train_graphs.json"
    output_path = tmp_path / "train_pairs.json"
    config_path = tmp_path / "effective_training_config.json"
    observed_model_names = []
    tasks_path.write_text(json.dumps(tiny_task_inputs()), encoding="utf-8")
    labels_path.write_text(json.dumps(tiny_labels()), encoding="utf-8")
    graphs_path.write_text(json.dumps(tiny_graphs()), encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "method": "dense_rgcn_graph_retriever",
                "profile": "full",
                "encoder": {
                    "model": "models/local-e5",
                    "query_prefix": "query: ",
                    "passage_prefix": "passage: ",
                },
                "pair_sampling": {
                    "random_seed": 7,
                    "easy_random_per_positive": 0,
                    "hard_bm25_per_positive": 0,
                    "hard_dense_per_positive": 1,
                    "hard_graph_neighbor_per_positive": 0,
                    "hard_pool_size": 10,
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeDenseTaskRetriever:
        def __init__(
            self,
            model_name="intfloat/e5-base-v2",
            batch_size=64,
            query_prefix="query: ",
            passage_prefix="passage: ",
            encoder=None,
        ):
            observed_model_names.append(model_name)

        def rank(self, task_input):
            return [
                pair_data.RankedNode(node_id=memory_item["id"], score=float(index))
                for index, memory_item in enumerate(task_input["memory_items"])
            ]

    monkeypatch.setattr(pair_data, "DenseTaskRetriever", FakeDenseTaskRetriever)

    exit_code = build_train_pairs_main(
        [
            "--tasks",
            str(tasks_path),
            "--labels",
            str(labels_path),
            "--graphs",
            str(graphs_path),
            "--output",
            str(output_path),
            "--config",
            str(config_path),
        ]
    )

    assert exit_code == 0
    assert observed_model_names == ["models/local-e5"]
