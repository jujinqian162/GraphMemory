from __future__ import annotations

import sys
from pathlib import Path

from graph_memory.config import CONFIG_LOADER
from graph_memory.registry import Registry
from graph_memory.registry.stage_configs import ImportanceStageConfig, RetrieveStageConfig
from scripts.workflow.manifest import initialize_experiment, load_experiment_config
from scripts.workflow.planner import build_stage_plan, required_stages_for_methods
from scripts.workflow.registry import get_workflow, validate_workflow_registry
from scripts.workflow.types import StageId, WorkflowId
from scripts.workflow.workflows import MEMORY_STREAM_WORKFLOW

MEMORY_STREAM = "memory_stream"


def test_memory_stream_workflow_generates_run_local_importance_stage_config(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "memory-stream",
        config=load_experiment_config(),
        run_root=tmp_path,
        profile="smoke",
        methods=[MEMORY_STREAM],
        force=True,
    )

    assert manifest["artifacts"]["importance"][MEMORY_STREAM] == {
        "scores": (tmp_path / "memory-stream" / "importance" / "test.memory_stream.importance.json").as_posix(),
        "run_summary": (
            tmp_path / "memory-stream" / "importance" / "test.memory_stream.importance.run_summary.json"
        ).as_posix(),
    }
    stage_path = Path(manifest["stage_configs"]["importance"][MEMORY_STREAM])
    assert stage_path == tmp_path / "memory-stream" / "config" / "stages" / "importance" / "memory_stream.json"
    assert stage_path.is_file()

    config = CONFIG_LOADER.load(Registry.configs.IMPORTANCE, ["--config", str(stage_path)])
    assert isinstance(config, ImportanceStageConfig)
    assert config.io.tasks == Path(manifest["artifacts"]["inputs"]["test"]["input"])
    assert config.io.output == Path(manifest["artifacts"]["importance"][MEMORY_STREAM]["scores"])
    assert config.io.summary == Path(manifest["artifacts"]["importance"][MEMORY_STREAM]["run_summary"])
    assert config.io.cache_dir == Path(manifest["effective_config"]["memory_stream"]["annotation"]["cache_dir"])
    assert config.job.model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert config.job.tp_plan is None
    assert "io" not in manifest["effective_config"]["memory_stream"]
    assert {"tasks", "output", "summary"}.isdisjoint(
        manifest["effective_config"]["memory_stream"]["annotation"]
    )
    assert not Path("configs/stages").exists()


def test_memory_stream_importance_command_uses_generated_stage_config(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "memory-stream-plan",
        config=load_experiment_config(),
        run_root=tmp_path,
        profile="smoke",
        methods=[MEMORY_STREAM],
        force=True,
    )

    commands = build_stage_plan(manifest, stages=["importance"], methods=[MEMORY_STREAM])

    assert len(commands) == 1
    command = commands[0]
    assert command.stage is StageId.IMPORTANCE
    assert command.method == MEMORY_STREAM
    assert command.argv == [
        sys.executable,
        "scripts/annotate_importance.py",
        "--config",
        manifest["stage_configs"]["importance"][MEMORY_STREAM],
    ]
    assert "configs/stages" not in command.argv[3].replace("\\", "/")


def test_memory_stream_workflow_is_registered_without_train_stages() -> None:
    validate_workflow_registry()

    assert get_workflow(MEMORY_STREAM) is MEMORY_STREAM_WORKFLOW
    assert MEMORY_STREAM_WORKFLOW.identifier is WorkflowId.MEMORY_STREAM_RETRIEVAL
    assert required_stages_for_methods([MEMORY_STREAM]) == [
        "prepare",
        "graphs",
        "importance",
        "retrieve",
        "evaluate",
        "aggregate",
    ]
    assert [step.stage for step in MEMORY_STREAM_WORKFLOW.steps] == [
        StageId.PREPARE,
        StageId.GRAPHS,
        StageId.IMPORTANCE,
        StageId.RETRIEVE,
        StageId.EVALUATE,
        StageId.AGGREGATE,
    ]


def test_memory_stream_retrieve_config_consumes_manifest_importance_path(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "memory-stream-retrieve-config",
        config=load_experiment_config(),
        run_root=tmp_path,
        profile="smoke",
        methods=[MEMORY_STREAM],
        force=True,
    )

    retrieve_config = CONFIG_LOADER.load(
        Registry.configs.RETRIEVE,
        ["--config", manifest["stage_configs"]["retrieve"][MEMORY_STREAM]],
    )

    assert isinstance(retrieve_config, RetrieveStageConfig)
    assert retrieve_config.io.importance == Path(manifest["artifacts"]["importance"][MEMORY_STREAM]["scores"])
    assert retrieve_config.io.graphs is None
    assert retrieve_config.job.method.value == MEMORY_STREAM
