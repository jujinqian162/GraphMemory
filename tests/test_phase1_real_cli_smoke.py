from __future__ import annotations

import json
from pathlib import Path

from graph_memory.io import read_json
import scripts.aggregate_tables as aggregate_tables
import scripts.build_graphs as build_graphs
import scripts.evaluate_retrieval as evaluate_retrieval
import scripts.prepare_hotpotqa as prepare_hotpotqa
import scripts.run_retrieval as run_retrieval


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
