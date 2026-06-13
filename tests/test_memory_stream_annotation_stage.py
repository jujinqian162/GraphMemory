from __future__ import annotations

from pathlib import Path

import pytest

from graph_memory.config import CONFIG_LOADER
from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.io import read_json, write_json
from graph_memory.registry import Registry
from graph_memory.registry.ids import StageId
from graph_memory.registry.stage_configs import (
    ImportanceAnnotationSettings,
    ImportanceIO,
    ImportanceStageConfig,
)
from graph_memory.retrieval.methods.memory_stream.contracts import GenerationResult
from graph_memory.validation import ContractValidationError, validate_importance_artifact
from scripts import annotate_importance as annotate_script


def _tasks() -> list[MemoryTaskInput]:
    return [
        {
            "task_id": "hotpot_ms_1",
            "query": "Which river runs through Paris?",
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
                    "text": "The Seine runs through Paris.",
                    "source": "Paris",
                    "sentence_id": 0,
                    "position": 1,
                },
            ],
        }
    ]


def _config(tmp_path: Path) -> ImportanceStageConfig:
    return ImportanceStageConfig(
        io=ImportanceIO(
            tasks=tmp_path / "test_memory_tasks.input.json",
            output=tmp_path / "importance" / "test.memory_stream.importance.json",
            summary=tmp_path / "importance" / "test.memory_stream.importance.run_summary.json",
            cache_dir=tmp_path / "cache" / "memory_stream_importance",
        ),
        job=ImportanceAnnotationSettings(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            model_path=Path("models/Qwen2.5-7B-Instruct"),
            prompt_version="memory-stream-importance-v1",
            device="auto",
            trust_remote_code=True,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            tp_plan=None,
            do_sample=False,
            use_cache=True,
            max_new_tokens=256,
        ),
    )


def _write_stage_config(path: Path, config: ImportanceStageConfig) -> None:
    write_json(path, CONFIG_LOADER.to_json(config))


class FakeLocalTransformersImportanceRuntime:
    instances: list["FakeLocalTransformersImportanceRuntime"] = []
    response = '{"scores":{"m0":8,"m1":4}}'

    def __init__(self, settings: ImportanceAnnotationSettings) -> None:
        self.settings = settings
        self.load_calls = 0
        self.generate_calls = 0
        self.__class__.instances.append(self)

    def load(self) -> dict[str, object]:
        self.load_calls += 1
        return {"model_load_seconds": 0.125, "device": "fake"}

    def generate(
        self,
        messages: list[dict[str, str]],
        settings: ImportanceAnnotationSettings,
    ) -> GenerationResult:
        _ = (messages, settings)
        self.generate_calls += 1
        return GenerationResult(
            text=self.__class__.response,
            generated_tokens=6,
            generation_seconds=0.25,
        )


def test_importance_stage_config_registry_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "importance.json"
    expected = _config(tmp_path)

    _write_stage_config(path, expected)

    assert Registry.configs.IMPORTANCE.stage is StageId.IMPORTANCE
    assert CONFIG_LOADER.load(Registry.configs.IMPORTANCE, ["--config", str(path)]) == expected


def test_repository_eval_1000_importance_config_is_loadable() -> None:
    config_path = Path("configs/stages/memory_stream_importance_eval_1000.json")

    config = CONFIG_LOADER.load(Registry.configs.IMPORTANCE, ["--config", str(config_path)])

    assert config.io.tasks == Path("data/hotpotqa/processed/phase2_baselines/eval_1000.input.json")
    assert config.io.output == Path(
        "runs/memory_stream_importance_eval1000/importance/test.memory_stream.importance.json"
    )
    assert config.io.summary == Path(
        "runs/memory_stream_importance_eval1000/importance/test.memory_stream.importance.run_summary.json"
    )
    assert config.io.cache_dir == Path("data/cache/memory_stream_importance")
    assert config.job.model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert config.job.model_path == Path("models/Qwen2.5-7B-Instruct")


def test_annotate_importance_cli_writes_final_artifact_and_run_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config_path = tmp_path / "importance.stage.json"
    write_json(config.io.tasks, _tasks())
    _write_stage_config(config_path, config)
    FakeLocalTransformersImportanceRuntime.instances = []
    FakeLocalTransformersImportanceRuntime.response = '{"scores":{"m0":8,"m1":4}}'
    monkeypatch.setattr(
        "graph_memory.stages.importance.LocalTransformersImportanceRuntime",
        FakeLocalTransformersImportanceRuntime,
    )

    exit_code = annotate_script.main(["--config", str(config_path)])

    artifact = read_json(config.io.output)
    summary = read_json(config.io.summary)
    assert exit_code == 0
    validate_importance_artifact(artifact, _tasks())
    assert artifact["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert artifact["tasks"][0]["scores"] == {"m0": 8, "m1": 4}
    assert summary["script"] == "annotate_importance.py"
    assert summary["status"] == "success"
    assert summary["counts"]["tasks"] == 1
    assert summary["counts"]["memory_items"] == 2
    assert summary["counts"]["cache_hits"] == 0
    assert summary["counts"]["model_load_count"] == 1
    assert summary["counts"]["generation_calls"] == 1
    assert summary["counts"]["generated_tokens"] == 6
    assert summary["timings"]["model_load_seconds"] == pytest.approx(0.125)
    assert summary["timings"]["generation_seconds"] == pytest.approx(0.25)
    assert summary["effective_config"]["model_path"] == "models/Qwen2.5-7B-Instruct"
    assert summary["outputs"]["importance"] == str(config.io.output)
    assert len(FakeLocalTransformersImportanceRuntime.instances) == 1


def test_annotate_importance_cli_reuses_cache_without_loading_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config_path = tmp_path / "importance.stage.json"
    write_json(config.io.tasks, _tasks())
    _write_stage_config(config_path, config)
    FakeLocalTransformersImportanceRuntime.instances = []
    monkeypatch.setattr(
        "graph_memory.stages.importance.LocalTransformersImportanceRuntime",
        FakeLocalTransformersImportanceRuntime,
    )

    assert annotate_script.main(["--config", str(config_path)]) == 0
    assert annotate_script.main(["--config", str(config_path)]) == 0

    summary = read_json(config.io.summary)
    assert summary["counts"]["cache_hits"] == 1
    assert summary["counts"]["model_load_count"] == 0
    assert summary["counts"]["generation_calls"] == 0
    assert len(FakeLocalTransformersImportanceRuntime.instances) == 1


def test_annotate_importance_cli_writes_failed_summary_without_final_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config_path = tmp_path / "importance.stage.json"
    write_json(config.io.tasks, _tasks())
    _write_stage_config(config_path, config)
    FakeLocalTransformersImportanceRuntime.instances = []
    FakeLocalTransformersImportanceRuntime.response = '{"scores":{"m0":true,"m1":4}}'
    monkeypatch.setattr(
        "graph_memory.stages.importance.LocalTransformersImportanceRuntime",
        FakeLocalTransformersImportanceRuntime,
    )

    with pytest.raises(ContractValidationError, match="m0.*integer"):
        annotate_script.main(["--config", str(config_path)])

    assert not config.io.output.exists()
    summary = read_json(config.io.summary)
    assert summary["status"] == "failed"
    assert "m0" in summary["error"]


def test_annotate_importance_environment_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.setenv(key, "sentinel")
    monkeypatch.setenv("ACCELERATE_USE_DEEPSPEED", "true")

    annotate_script._prepare_local_transformers_environment()

    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        assert key not in annotate_script.os.environ
    assert annotate_script.os.environ["ACCELERATE_USE_DEEPSPEED"] == "false"


def test_failed_run_does_not_overwrite_existing_successful_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config_path = tmp_path / "importance.stage.json"
    write_json(config.io.tasks, _tasks())
    _write_stage_config(config_path, config)
    FakeLocalTransformersImportanceRuntime.instances = []
    FakeLocalTransformersImportanceRuntime.response = '{"scores":{"m0":8,"m1":4}}'
    monkeypatch.setattr(
        "graph_memory.stages.importance.LocalTransformersImportanceRuntime",
        FakeLocalTransformersImportanceRuntime,
    )
    assert annotate_script.main(["--config", str(config_path)]) == 0
    original_artifact_text = config.io.output.read_text(encoding="utf-8")

    cache_files = list(config.io.cache_dir.rglob("*.json"))
    assert cache_files
    for cache_file in cache_files:
        cache_file.unlink()
    FakeLocalTransformersImportanceRuntime.response = '{"scores":{"m0":8}}'

    with pytest.raises(ContractValidationError):
        annotate_script.main(["--config", str(config_path)])

    assert config.io.output.read_text(encoding="utf-8") == original_artifact_text
    assert read_json(config.io.summary)["status"] == "failed"
