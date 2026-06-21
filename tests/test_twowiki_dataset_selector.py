from __future__ import annotations

from pathlib import Path

from graph_memory.config import CONFIG_LOADER
from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.ranking import RankedResult
from graph_memory.io import read_json, write_json
from graph_memory.registry import Registry
from graph_memory.registry.retrieval import Bm25RetrievalSettings
from graph_memory.registry.stage_configs import EvaluateIO, EvaluateStageConfig, RetrieveIO, RetrieveStageConfig
from scripts import build_graphs, evaluate_retrieval, run_retrieval, tune_graph_rerank


def _twowiki_task() -> dict[str, object]:
    return {
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
                "title": "Ada Lovelace",
                "sentence_index": 0,
                "position": 1,
                "text": "Ada was the daughter of Beth.",
            },
        ],
        "metadata": {"dataset": "2wiki", "raw_id": "abc123"},
    }


def _twowiki_label() -> dict[str, object]:
    return {
        "task_id": "2wiki_abc123",
        "gold_answer": "Beth",
        "gold_evidence_sentence_ids": ["m0", "m1"],
        "gold_dependency_edges": [["m0", "m1"]],
        "metadata": {
            "question_type": "compositional",
            "path_label_source": "evidences",
            "path_supported": True,
            "mapping_ambiguity_count": 0,
        },
    }


def test_stage_config_dataset_defaults_to_hotpotqa_for_old_configs(tmp_path: Path) -> None:
    config_path = tmp_path / "retrieve.json"
    write_json(
        config_path,
        {
            "io": {
                "tasks": "tasks.json",
                "graphs": None,
                "output": "predictions.json",
                "summary": "predictions.run_summary.json",
            },
            "job": {"method": "bm25", "top_k": 3},
        },
    )

    config = CONFIG_LOADER.load(Registry.configs.RETRIEVE, ["--config", str(config_path)])

    assert config.dataset == "hotpotqa"


def test_build_graphs_script_accepts_twowiki_dataset_selector(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.json"
    graphs_path = tmp_path / "graphs.json"
    write_json(tasks_path, [_twowiki_task()])

    assert build_graphs.main(["--dataset", "twowiki", "--input", str(tasks_path), "--output", str(graphs_path)]) == 0

    graphs = read_json(graphs_path)
    assert graphs[0]["task_id"] == "2wiki_abc123"
    assert graphs[0]["nodes"][1]["metadata"]["question_type"] == "compositional"


def test_run_retrieval_script_uses_twowiki_dataset_selector(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.json"
    output_path = tmp_path / "predictions.json"
    summary_path = tmp_path / "predictions.run_summary.json"
    config_path = tmp_path / "retrieve.json"
    write_json(tasks_path, [_twowiki_task()])
    write_json(
        config_path,
        CONFIG_LOADER.to_json(
            RetrieveStageConfig(
                io=RetrieveIO(tasks=tasks_path, graphs=None, output=output_path, summary=summary_path),
                job=Bm25RetrievalSettings(top_k=2),
                dataset="twowiki",
            )
        ),
    )

    assert run_retrieval.main(["--config", str(config_path)]) == 0

    predictions = read_json(output_path)
    summary = read_json(summary_path)
    assert predictions[0]["task_id"] == "2wiki_abc123"
    assert predictions[0]["method"] == "bm25"
    assert summary["effective_config"]["dataset"] == "twowiki"


def test_evaluate_retrieval_script_uses_twowiki_labels_and_path_metrics(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.json"
    labels_path = tmp_path / "labels.json"
    graphs_path = tmp_path / "graphs.json"
    metrics_path = tmp_path / "metrics.csv"
    config_path = tmp_path / "evaluate.json"
    prediction: RankedResult = {
        "task_id": "2wiki_abc123",
        "method": "dense_graph_rerank",
        "ranked_nodes": [{"node_id": "m0", "score": 2.0}, {"node_id": "m1", "score": 1.0}],
        "retrieved_subgraph": {
            "nodes": ["m0", "m1"],
            "edges": [{"source": "m0", "target": "m1", "edge_type": "bridge", "weight": 1.0, "directed": False}],
        },
        "latency_ms": 1.0,
        "input_tokens": 8,
    }
    graph: MemoryGraph = {
        "task_id": "2wiki_abc123",
        "nodes": [
            {"id": "q", "node_type": "question", "text": "Who?"},
            {"id": "m0", "node_type": "graph_item", "node_kind": "document_sentence", "text": "Film A."},
            {"id": "m1", "node_type": "graph_item", "node_kind": "document_sentence", "text": "Ada."},
        ],
        "edges": [{"source": "m0", "target": "m1", "edge_type": "bridge", "weight": 1.0, "directed": False}],
    }
    write_json(predictions_path, [prediction])
    write_json(labels_path, [_twowiki_label()])
    write_json(graphs_path, [graph])
    write_json(
        config_path,
        CONFIG_LOADER.to_json(
            EvaluateStageConfig(
                io=EvaluateIO(
                    predictions=predictions_path,
                    labels=labels_path,
                    graphs=graphs_path,
                    output=metrics_path,
                ),
                dataset="twowiki",
            )
        ),
    )

    assert evaluate_retrieval.main(["--config", str(config_path)]) == 0

    metrics = metrics_path.read_text(encoding="utf-8")
    assert "dense_graph_rerank" in metrics
    assert ",1.0,1.0,1.0," in metrics


def test_tune_graph_rerank_script_uses_twowiki_dataset_selector(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.json"
    labels_path = tmp_path / "labels.json"
    graphs_path = tmp_path / "graphs.json"
    grid_path = tmp_path / "grid.json"
    selected_config_path = tmp_path / "selected.json"
    write_json(tasks_path, [_twowiki_task()])
    write_json(labels_path, [_twowiki_label()])
    write_json(
        grid_path,
        {
            "lambda_init": [1.0],
            "lambda_query": [0.0],
            "lambda_neighbor": [0.0],
            "lambda_bridge": [0.0],
            "lambda_path": [0.0],
            "seed_top_s": [1],
            "max_hops": [1],
            "neighbor_type_weights": {
                "sequential": 1.0,
                "entity_overlap": 1.0,
                "bridge": 1.0,
            },
        },
    )

    assert build_graphs.main(["--dataset", "twowiki", "--input", str(tasks_path), "--output", str(graphs_path)]) == 0
    assert tune_graph_rerank.main(
        [
            "--dataset",
            "twowiki",
            "--method",
            "bm25_graph_rerank",
            "--tasks",
            str(tasks_path),
            "--labels",
            str(labels_path),
            "--graphs",
            str(graphs_path),
            "--output_config",
            str(selected_config_path),
            "--top_k",
            "2",
            "--grid_config",
            str(grid_path),
        ]
    ) == 0

    selected_config = read_json(selected_config_path)
    summary = read_json(selected_config_path.with_name("selected.run_summary.json"))
    assert "lambda_init" in selected_config
    assert summary["effective_config"]["dataset"] == "twowiki"
