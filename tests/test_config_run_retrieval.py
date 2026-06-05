from __future__ import annotations

import json
from pathlib import Path

from graph_memory.io import read_json
from scripts import run_retrieval
from tests.test_phase1_real_retrieval import retrieval_task_inputs


def test_run_retrieval_script_loads_stage_config_and_preserves_artifact_io(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.json"
    output_path = tmp_path / "predictions.json"
    tasks_path.write_text(json.dumps(retrieval_task_inputs()), encoding="utf-8")

    exit_code = run_retrieval.main(
        [
            "--method",
            "bm25",
            "--tasks",
            str(tasks_path),
            "--output",
            str(output_path),
            "--top_k",
            "2",
        ]
    )

    predictions = read_json(output_path)
    summary = read_json(output_path.with_name("predictions.run_summary.json"))

    assert exit_code == 0
    assert predictions[0]["method"] == "bm25"
    assert summary["status"] == "success"
    assert summary["effective_config"]["method"] == "bm25"
    assert summary["outputs"]["predictions"] == str(output_path)


def test_run_retrieval_summary_preserves_public_encoder_cli_values_for_bm25(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.json"
    output_path = tmp_path / "predictions.json"
    tasks_path.write_text(json.dumps(retrieval_task_inputs()), encoding="utf-8")

    run_retrieval.main(
        [
            "--method",
            "bm25",
            "--tasks",
            str(tasks_path),
            "--output",
            str(output_path),
            "--encoder_model",
            "custom-model",
            "--query_prefix",
            "custom query: ",
            "--passage_prefix",
            "custom passage: ",
        ]
    )

    summary = read_json(output_path.with_name("predictions.run_summary.json"))

    assert summary["effective_config"]["encoder_model"] == "custom-model"
    assert summary["effective_config"]["query_prefix"] == "custom query: "
    assert summary["effective_config"]["passage_prefix"] == "custom passage: "


def test_run_retrieval_script_uses_config_loader_and_stage_runner() -> None:
    source = Path("scripts/run_retrieval.py").read_text(encoding="utf-8")

    assert "CONFIG_LOADER.load(Registry.configs.RETRIEVE" in source
    assert "run_retrieve_stage(" in source
    assert "DenseRuntime" not in source
    assert "TrainableGraphRuntime" not in source
