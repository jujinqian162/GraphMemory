from __future__ import annotations

from pathlib import Path

from graph_memory.config import CONFIG_LOADER
from graph_memory.io import read_json, write_json
from graph_memory.registry.retrieval import Bm25RetrievalSettings
from graph_memory.registry.stage_configs import RetrieveIO, RetrieveStageConfig
from scripts import run_retrieval
from tests.test_phase1_real_retrieval import retrieval_task_inputs


def test_run_retrieval_script_loads_complete_stage_config_and_preserves_io(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.json"
    output_path = tmp_path / "predictions.json"
    summary_path = tmp_path / "predictions.run_summary.json"
    config_path = tmp_path / "retrieve.json"
    write_json(tasks_path, retrieval_task_inputs())
    write_json(
        config_path,
        CONFIG_LOADER.to_json(
            RetrieveStageConfig(
                io=RetrieveIO(
                    tasks=tasks_path,
                    graphs=None,
                    output=output_path,
                    summary=summary_path,
                ),
                job=Bm25RetrievalSettings(top_k=2),
            )
        ),
    )

    exit_code = run_retrieval.main(["--config", str(config_path)])
    predictions = read_json(output_path)
    summary = read_json(summary_path)

    assert exit_code == 0
    assert predictions[0]["method"] == "bm25"
    assert summary["status"] == "success"
    assert summary["effective_config"]["method"] == "bm25"
    assert summary["effective_config"]["provenance"]["encoder"] is None
    assert summary["outputs"]["predictions"] == str(output_path)


def test_run_retrieval_script_uses_config_loader_stage_runner_and_runtime_provenance() -> None:
    source = Path("scripts/run_retrieval.py").read_text(encoding="utf-8")

    assert "CONFIG_LOADER.load(Registry.configs.RETRIEVE" in source
    assert "run_retrieve_stage(" in source
    assert "_encoder_settings" not in source
    assert "_checkpoint_path" not in source
    assert "_device" not in source
