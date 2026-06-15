from __future__ import annotations

import hashlib
from pathlib import Path

from graph_memory.config import CONFIG_LOADER
from graph_memory.io import read_json, write_json
from graph_memory.registry.retrieval import Bm25RetrievalSettings, DenseEncoderSettings, MemoryStreamRetrievalSettings
from graph_memory.registry.stage_configs import RetrieveIO, RetrieveStageConfig
from graph_memory.retrieval.methods.memory_stream.artifact import importance_content_digest
from scripts import run_retrieval
from tests.test_phase1_real_retrieval import FakeEncoder, retrieval_task_inputs


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


def test_run_retrieval_script_serializes_memory_stream_provenance_and_settings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "graph_memory.retrieval.methods.flat.dense.load_sentence_transformer",
        lambda _model_name: FakeEncoder(),
    )
    tasks = retrieval_task_inputs()
    tasks_path = tmp_path / "tasks.json"
    output_path = tmp_path / "predictions.json"
    summary_path = tmp_path / "predictions.run_summary.json"
    config_path = tmp_path / "retrieve-memory-stream.json"
    importance_path = tmp_path / "dev.first_1000.importance.json"
    write_json(tasks_path, tasks)
    write_json(
        importance_path,
        {
            "schema_version": 1,
            "method": "memory_stream",
            "tasks": [
                {
                    "task_id": task["task_id"],
                    "content_digest": importance_content_digest(task),
                    "scores": {item["id"]: index + 1 for index, item in enumerate(task["memory_items"])},
                }
                for task in tasks
            ],
        },
    )
    config = RetrieveStageConfig(
        io=RetrieveIO(
            tasks=tasks_path,
            graphs=None,
            output=output_path,
            summary=summary_path,
            importance=importance_path,
        ),
        job=MemoryStreamRetrievalSettings(
            top_k=2,
            encoder=DenseEncoderSettings(
                model_name="fake-model",
                query_prefix="query: ",
                passage_prefix="passage: ",
            ),
            recency_decay=1.0,
            capped_test_count=1,
        ),
    )
    write_json(config_path, CONFIG_LOADER.to_json(config))

    assert run_retrieval.main(["--config", str(config_path)]) == 0
    predictions = read_json(output_path)
    summary = read_json(summary_path)
    expected_sha256 = hashlib.sha256(importance_path.read_bytes()).hexdigest()

    assert predictions[0]["method"] == "memory_stream"
    assert summary["effective_config"]["provenance"]["importance"] == {
        "path": str(importance_path),
        "sha256": expected_sha256,
        "schema_version": 1,
    }
    assert summary["effective_config"]["job"]["capped_test_count"] == 1
    assert summary["effective_config"]["job"]["relevance_weight"] == 1.0
    assert summary["effective_config"]["job"]["encoder"]["model_name"] == "fake-model"
