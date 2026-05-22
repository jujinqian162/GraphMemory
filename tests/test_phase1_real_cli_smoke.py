from __future__ import annotations

import json
from pathlib import Path

import pytest

from graph_memory.io import read_json
import scripts.aggregate_tables as aggregate_tables
import scripts.build_graphs as build_graphs
import scripts.evaluate_retrieval as evaluate_retrieval
import scripts.prepare_hotpotqa as prepare_hotpotqa
import scripts.run_retrieval as run_retrieval
import scripts.tune_graph_rerank as tune_graph_rerank


FIXTURE = Path(__file__).parent / "fixtures" / "hotpotqa_smoke.json"


def test_phase1_cli_pipeline_writes_contract_artifacts(tmp_path):
    task_inputs_path = tmp_path / "test_memory_tasks.input.json"
    labels_path = tmp_path / "test_memory_tasks.labels.json"
    combined_path = tmp_path / "test_memory_tasks.json"
    graphs_path = tmp_path / "test_graphs.json"
    predictions_path = tmp_path / "main_results_bm25.predictions.json"
    metrics_path = tmp_path / "main_results_bm25.csv"
    failures_path = tmp_path / "failure_cases_bm25.jsonl"
    aggregate_main_path = tmp_path / "main_results.csv"
    aggregate_path_path = tmp_path / "path_results.csv"
    aggregate_efficiency_path = tmp_path / "efficiency_results.csv"

    assert prepare_hotpotqa.main(
        [
            "--input",
            str(FIXTURE),
            "--output_input",
            str(task_inputs_path),
            "--output_labels",
            str(labels_path),
            "--output_combined",
            str(combined_path),
            "--max_examples",
            "1",
            "--seed",
            "13",
            "--offset",
            "0",
        ]
    ) == 0
    assert build_graphs.main(
        [
            "--input",
            str(task_inputs_path),
            "--output",
            str(graphs_path),
            "--max_query_overlap",
            "20",
            "--max_entity_neighbors",
            "10",
            "--max_bridge_edges",
            "50",
        ]
    ) == 0
    assert run_retrieval.main(
        [
            "--method",
            "bm25",
            "--tasks",
            str(task_inputs_path),
            "--output",
            str(predictions_path),
            "--top_k",
            "10",
        ]
    ) == 0
    assert evaluate_retrieval.main(
        [
            "--pred",
            str(predictions_path),
            "--labels",
            str(labels_path),
            "--graphs",
            str(graphs_path),
            "--output",
            str(metrics_path),
            "--failure_cases_output",
            str(failures_path),
            "--failure_case_limit",
            "5",
        ]
    ) == 0
    assert aggregate_tables.main(
        [
            "--input_dir",
            str(tmp_path),
            "--output_main",
            str(aggregate_main_path),
            "--output_path",
            str(aggregate_path_path),
            "--output_efficiency",
            str(aggregate_efficiency_path),
        ]
    ) == 0

    task_inputs = read_json(task_inputs_path)
    labels = read_json(labels_path)
    graphs = read_json(graphs_path)
    predictions = read_json(predictions_path)

    assert len(task_inputs) == len(labels) == len(graphs) == len(predictions) == 1
    assert predictions[0]["task_id"] == task_inputs[0]["task_id"] == labels[0]["task_id"]
    assert len(predictions[0]["ranked_nodes"]) == len(task_inputs[0]["memory_items"])

    input_and_graph_payload = json.dumps({"inputs": task_inputs, "graphs": graphs})
    assert "gold_answer" not in input_and_graph_payload
    assert "gold_evidence_nodes" not in input_and_graph_payload
    assert "supporting_facts" not in input_and_graph_payload

    assert "Method,Recall@2,Recall@5" in metrics_path.read_text(encoding="utf-8")
    assert "bm25" in aggregate_main_path.read_text(encoding="utf-8")
    assert aggregate_path_path.exists()
    assert aggregate_efficiency_path.exists()


def test_tune_graph_rerank_cli_reads_search_space_and_writes_neighbor_type_weights(tmp_path):
    task_inputs_path = tmp_path / "dev_memory_tasks.input.json"
    labels_path = tmp_path / "dev_memory_tasks.labels.json"
    graphs_path = tmp_path / "dev_graphs.json"
    grid_path = tmp_path / "graph_rerank.search_space.json"
    selected_config_path = tmp_path / "dense_graph_rerank.dev_selected.json"

    assert prepare_hotpotqa.main(
        [
            "--input",
            str(FIXTURE),
            "--output_input",
            str(task_inputs_path),
            "--output_labels",
            str(labels_path),
            "--max_examples",
            "1",
            "--seed",
            "13",
            "--offset",
            "0",
        ]
    ) == 0
    assert build_graphs.main(
        [
            "--input",
            str(task_inputs_path),
            "--output",
            str(graphs_path),
        ]
    ) == 0
    grid_path.write_text(
        json.dumps(
            {
                "lambda_init": [1.0],
                "lambda_query": [0.0],
                "lambda_neighbor": [0.0],
                "lambda_bridge": [0.0],
                "lambda_path": [0.0],
                "seed_top_s": [1],
                "max_hops": [1],
                "neighbor_type_weights": {
                    "sequential": 0.3,
                    "entity_overlap": 0.7,
                    "bridge": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )

    assert tune_graph_rerank.main(
        [
            "--method",
            "bm25_graph_rerank",
            "--tasks",
            str(task_inputs_path),
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
    candidate_rows = read_json(selected_config_path.with_name(f"{selected_config_path.stem}.candidates.json"))

    assert "neighbor_type_weights" in selected_config
    assert "type_weights" not in selected_config
    assert "query_overlap" not in selected_config["neighbor_type_weights"]
    assert "neighbor_type_weights" in candidate_rows[0]["config"]
    assert "type_weights" not in candidate_rows[0]["config"]


def test_aggregate_tables_includes_experiment_runner_metric_filenames(tmp_path):
    metrics_path = tmp_path / "test.dense.metrics.csv"
    metrics_path.write_text(
        "\n".join(
            [
                "Method,Recall@2,Recall@5,Recall@10,Evidence F1@5,Evidence F1@10,Full Support@5,Full Support@10,MRR,Connected Evidence Recall@5,Connected Evidence Recall@10,Query-Evidence Connectivity@10,Path Recall@10,Edge Recall@10,Retrieval Latency / Query,Index Build Time,Graph Construction Time,Memory Size,Avg Retrieved Nodes,Avg Retrieved Edges",
                "dense,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.11,N/A,N/A,12.3,0.0,0.0,42.0,10.0,0.0",
            ]
        ),
        encoding="utf-8",
    )
    aggregate_main_path = tmp_path / "main_results.csv"
    aggregate_path_path = tmp_path / "path_results.csv"
    aggregate_efficiency_path = tmp_path / "efficiency_results.csv"

    assert aggregate_tables.main(
        [
            "--input_dir",
            str(tmp_path),
            "--output_main",
            str(aggregate_main_path),
            "--output_path",
            str(aggregate_path_path),
            "--output_efficiency",
            str(aggregate_efficiency_path),
        ]
    ) == 0

    assert "dense,0.1,0.2,0.3" in aggregate_main_path.read_text(encoding="utf-8")


def test_prepare_hotpotqa_drops_invalid_examples_before_sampling(tmp_path):
    raw_path = tmp_path / "raw.json"
    valid_first = {
        "_id": "valid-first",
        "question": "Where is the Eiffel Tower?",
        "answer": "Paris",
        "context": [["Eiffel Tower", ["The Eiffel Tower is in Paris."]]],
        "supporting_facts": [["Eiffel Tower", 0]],
    }
    malformed = {
        "question": "Missing id",
        "answer": "nowhere",
        "context": [["Missing", ["This record has no id."]]],
        "supporting_facts": [["Missing", 0]],
    }
    unconvertible = {
        "_id": "bad-support",
        "question": "Which support is missing?",
        "answer": "missing",
        "context": [["Known", ["Only this sentence exists."]]],
        "supporting_facts": [["Unknown", 0]],
    }
    valid_second = {
        "_id": "valid-second",
        "question": "Where is the Louvre?",
        "answer": "Paris",
        "context": [["Louvre", ["The Louvre is in Paris."]]],
        "supporting_facts": [["Louvre", 0]],
    }
    raw_path.write_text(json.dumps([valid_first, malformed, unconvertible, valid_second]), encoding="utf-8")

    task_inputs_path = tmp_path / "memory_tasks.input.json"
    labels_path = tmp_path / "memory_tasks.labels.json"

    assert prepare_hotpotqa.main(
        [
            "--input",
            str(raw_path),
            "--output_input",
            str(task_inputs_path),
            "--output_labels",
            str(labels_path),
            "--max_examples",
            "2",
            "--seed",
            "13",
            "--offset",
            "0",
        ]
    ) == 0

    task_inputs = read_json(task_inputs_path)
    summary = read_json(tmp_path / "memory_tasks.input.run_summary.json")

    assert {task_input["task_id"] for task_input in task_inputs} == {"hotpot_valid-first", "hotpot_valid-second"}
    assert summary["counts"]["raw_examples"] == 4
    assert summary["counts"]["valid_examples"] == 2
    assert summary["counts"]["invalid_examples_dropped"] == 2
    assert summary["counts"]["selected_examples"] == 2
    assert any("_id" in reason for reason in summary["counts"]["invalid_example_reasons"])
    assert any("supporting fact" in reason for reason in summary["counts"]["invalid_example_reasons"])


def test_prepare_hotpotqa_drops_examples_that_fail_output_validation(tmp_path):
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(
        json.dumps(
            [
                {
                    "_id": "empty-text",
                    "question": "Which sentence is empty?",
                    "answer": "empty",
                    "context": [["Empty", [""]]],
                    "supporting_facts": [["Empty", 0]],
                },
                {
                    "_id": "valid",
                    "question": "Where is the Eiffel Tower?",
                    "answer": "Paris",
                    "context": [["Eiffel Tower", ["The Eiffel Tower is in Paris."]]],
                    "supporting_facts": [["Eiffel Tower", 0]],
                },
            ]
        ),
        encoding="utf-8",
    )

    task_inputs_path = tmp_path / "memory_tasks.input.json"
    labels_path = tmp_path / "memory_tasks.labels.json"

    assert prepare_hotpotqa.main(
        [
            "--input",
            str(raw_path),
            "--output_input",
            str(task_inputs_path),
            "--output_labels",
            str(labels_path),
        ]
    ) == 0

    task_inputs = read_json(task_inputs_path)
    summary = read_json(tmp_path / "memory_tasks.input.run_summary.json")

    assert [task_input["task_id"] for task_input in task_inputs] == ["hotpot_valid"]
    assert summary["counts"]["valid_examples"] == 1
    assert summary["counts"]["invalid_examples_dropped"] == 1
    assert any("field=text" in reason for reason in summary["counts"]["invalid_example_reasons"])


def test_prepare_hotpotqa_strict_mode_fails_on_invalid_example(tmp_path):
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(
        json.dumps(
            [
                {
                    "_id": "valid",
                    "question": "Where is the Eiffel Tower?",
                    "answer": "Paris",
                    "context": [["Eiffel Tower", ["The Eiffel Tower is in Paris."]]],
                    "supporting_facts": [["Eiffel Tower", 0]],
                },
                {
                    "_id": "bad-support",
                    "question": "Which support is missing?",
                    "answer": "missing",
                    "context": [["Known", ["Only this sentence exists."]]],
                    "supporting_facts": [["Unknown", 0]],
                },
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="index=1"):
        prepare_hotpotqa.main(
            [
                "--input",
                str(raw_path),
                "--output_input",
                str(tmp_path / "memory_tasks.input.json"),
                "--output_labels",
                str(tmp_path / "memory_tasks.labels.json"),
                "--strict_invalid_examples",
            ]
        )
