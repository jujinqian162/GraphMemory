from __future__ import annotations

from pathlib import Path

import pytest

from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.io import read_json, write_json
from graph_memory.retrieval.methods.memory_stream.contracts import (
    GenerationResult,
    ImportanceMessage,
    ImportanceSettings,
)
from graph_memory.registry import Registry
from graph_memory.registry.retrieval import RetrievalMethodId
from graph_memory.validation import ContractValidationError
from scripts.workflow.types import StageId
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


class FakeLocalTransformersImportanceRuntime:
    instances: list["FakeLocalTransformersImportanceRuntime"] = []
    response = '{"scores":[8,4]}'

    def __init__(self, settings: ImportanceSettings) -> None:
        self.settings = settings
        self.load_calls = 0
        self.generate_calls = 0
        self.__class__.instances.append(self)

    def load(self) -> dict[str, object]:
        self.load_calls += 1
        return {"model_load_seconds": 0.125, "device": "fake"}

    def generate(
        self,
        messages: list[ImportanceMessage],
        settings: ImportanceSettings,
    ) -> GenerationResult:
        _ = (messages, settings)
        self.generate_calls += 1
        return GenerationResult(
            text=self.__class__.response,
            generated_tokens=6,
            generation_seconds=0.25,
        )


def test_annotate_importance_zero_argument_defaults() -> None:
    args = annotate_script.parse_args([])

    assert args.tasks == Path("data/hotpotqa/processed/dev_memory_tasks.input.json")
    assert args.output == Path("data/hotpotqa/processed/memory_stream/dev.importance.json")
    assert args.summary == Path("data/hotpotqa/processed/memory_stream/dev.importance.run_summary.json")
    assert args.cache_dir == Path("data/cache/memory_stream_importance")
    assert args.model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert args.model_path == Path("models/Qwen2.5-7B-Instruct")
    assert args.prompt_version == "memory-stream-importance-v2"
    assert args.device == "auto"
    assert args.max_new_tokens == 2048


def test_annotate_importance_derives_summary_from_custom_output() -> None:
    args = annotate_script.parse_args(["--output", "shared/custom.json"])

    assert args.output == Path("shared/custom.json")
    assert args.summary == Path("shared/custom.run_summary.json")


def test_importance_prepare_is_outside_workflow_and_method_registries() -> None:
    assert "importance" not in {stage.value for stage in StageId}
    assert "IMPORTANCE" not in vars(Registry.configs)
    assert "memory_stream" not in {method.value for method in RetrievalMethodId}


def test_annotation_module_imports_without_validation_preload() -> None:
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from graph_memory.retrieval.methods.memory_stream.annotation import annotate_importance_tasks",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_annotate_importance_zero_argument_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    write_json(Path("data/hotpotqa/processed/dev_memory_tasks.input.json"), _tasks())
    FakeLocalTransformersImportanceRuntime.instances = []
    FakeLocalTransformersImportanceRuntime.response = '{"scores":[8,4]}'

    exit_code = annotate_script.main(
        [],
        runtime_factory=lambda settings: FakeLocalTransformersImportanceRuntime(settings),
    )

    artifact_path = Path("data/hotpotqa/processed/memory_stream/dev.importance.json")
    summary_path = Path("data/hotpotqa/processed/memory_stream/dev.importance.run_summary.json")
    artifact = read_json(artifact_path)
    summary = read_json(summary_path)
    assert exit_code == 0
    assert artifact["tasks"][0]["scores"] == {"m0": 8, "m1": 4}
    assert summary["status"] == "success"
    assert summary["inputs"]["tasks"] == "data/hotpotqa/processed/dev_memory_tasks.input.json"
    assert summary["outputs"]["importance"] == "data/hotpotqa/processed/memory_stream/dev.importance.json"
    assert summary["counts"]["model_load_count"] == 1
    assert len(FakeLocalTransformersImportanceRuntime.instances) == 1


def test_annotate_importance_zero_argument_rerun_uses_cache_without_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    write_json(Path("data/hotpotqa/processed/dev_memory_tasks.input.json"), _tasks())
    FakeLocalTransformersImportanceRuntime.instances = []
    FakeLocalTransformersImportanceRuntime.response = '{"scores":[8,4]}'

    def runtime_factory(settings: ImportanceSettings) -> FakeLocalTransformersImportanceRuntime:
        return FakeLocalTransformersImportanceRuntime(settings)

    assert annotate_script.main([], runtime_factory=runtime_factory) == 0
    assert annotate_script.main([], runtime_factory=runtime_factory) == 0

    summary = read_json(
        Path("data/hotpotqa/processed/memory_stream/dev.importance.run_summary.json")
    )
    assert summary["counts"]["cache_hits"] == 1
    assert summary["counts"]["model_load_count"] == 0
    assert summary["counts"]["generation_calls"] == 0
    assert len(FakeLocalTransformersImportanceRuntime.instances) == 1


def test_annotate_importance_failed_rerun_preserves_successful_global_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    write_json(Path("data/hotpotqa/processed/dev_memory_tasks.input.json"), _tasks())
    FakeLocalTransformersImportanceRuntime.instances = []
    FakeLocalTransformersImportanceRuntime.response = '{"scores":[8,4]}'

    def runtime_factory(settings: ImportanceSettings) -> FakeLocalTransformersImportanceRuntime:
        return FakeLocalTransformersImportanceRuntime(settings)
    assert annotate_script.main([], runtime_factory=runtime_factory) == 0
    artifact_path = Path("data/hotpotqa/processed/memory_stream/dev.importance.json")
    original_artifact = artifact_path.read_text(encoding="utf-8")

    for cache_file in Path("data/cache/memory_stream_importance").rglob("*.json"):
        cache_file.unlink()
    FakeLocalTransformersImportanceRuntime.response = '{"scores":[8]}'

    with pytest.raises(ContractValidationError, match="expected=2.*observed=1"):
        annotate_script.main([], runtime_factory=runtime_factory)

    assert artifact_path.read_text(encoding="utf-8") == original_artifact
    summary = read_json(
        Path("data/hotpotqa/processed/memory_stream/dev.importance.run_summary.json")
    )
    assert summary["status"] == "failed"
    assert "expected=2 observed=1" in summary["error"]


def test_annotate_importance_environment_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.setenv(key, "sentinel")
    monkeypatch.setenv("ACCELERATE_USE_DEEPSPEED", "true")

    annotate_script._prepare_local_transformers_environment()

    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        assert key not in annotate_script.os.environ
    assert annotate_script.os.environ["ACCELERATE_USE_DEEPSPEED"] == "false"
