from __future__ import annotations

from pathlib import Path

from graph_memory.config import CONFIG_LOADER
from graph_memory.registry import Registry
from graph_memory.registry.methods import ArtifactKind, RetrievalLifecycle
from graph_memory.registry.retrieval import DenseFinetunedRetrievalSettings
from graph_memory.registry.stage_configs import DenseFinetuneTrainStageConfig, RetrieveStageConfig
from scripts.workflow.manifest import initialize_experiment, load_experiment_config
from scripts.workflow.planner import build_stage_plan
from scripts.workflow.registry import get_workflow, validate_workflow_registry
from scripts.workflow.types import StageId, WorkflowId
from scripts.workflow.workflows import DENSE_FT_WORKFLOW

DENSE_FT_METHOD = "dense_ft"


def test_dense_ft_workflow_is_selected_by_lifecycle() -> None:
    validate_workflow_registry()
    definition = Registry.methods.get(DENSE_FT_METHOD)

    assert definition.lifecycle is RetrievalLifecycle.DENSE_FINETUNE
    assert definition.train_artifact is not None
    assert definition.train_artifact.basename == "best_model"
    assert definition.train_artifact.kind is ArtifactKind.DIRECTORY
    assert get_workflow(DENSE_FT_METHOD) is DENSE_FT_WORKFLOW
    assert DENSE_FT_WORKFLOW.identifier is WorkflowId.DENSE_FINETUNE_RETRIEVAL
    assert [step.stage for step in DENSE_FT_WORKFLOW.steps] == [
        StageId.PREPARE,
        StageId.GRAPHS,
        StageId.PAIRS,
        StageId.TRAIN,
        StageId.RETRIEVE,
        StageId.EVALUATE,
        StageId.AGGREGATE,
    ]


def test_dense_ft_manifest_writes_model_directory_train_and_retrieve_configs(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "dense-ft",
        config=load_experiment_config(),
        run_root=tmp_path,
        profile="smoke",
        methods=[DENSE_FT_METHOD],
        force=True,
    )
    learned = manifest["artifacts"]["learned"][DENSE_FT_METHOD]
    train_config = CONFIG_LOADER.load(
        Registry.configs.TRAIN,
        ["--config", manifest["stage_configs"]["train"][DENSE_FT_METHOD]],
    )
    retrieve_config = CONFIG_LOADER.load(
        Registry.configs.RETRIEVE,
        ["--config", manifest["stage_configs"]["retrieve"][DENSE_FT_METHOD]],
    )

    assert isinstance(train_config, DenseFinetuneTrainStageConfig)
    assert train_config.io.model_dir == Path(learned["best_checkpoint"])
    assert train_config.io.model_dir.name == "best_model"
    assert isinstance(retrieve_config, RetrieveStageConfig)
    assert retrieve_config.io.graphs is None
    assert isinstance(retrieve_config.job, DenseFinetunedRetrievalSettings)
    assert retrieve_config.job.checkpoint == Path(learned["best_checkpoint"])


def test_dense_ft_plan_has_no_method_specific_argv_sprawl(tmp_path: Path) -> None:
    manifest = initialize_experiment(
        "dense-ft-plan",
        config=load_experiment_config(),
        run_root=tmp_path,
        profile="smoke",
        methods=[DENSE_FT_METHOD],
        force=True,
    )
    commands = build_stage_plan(
        manifest,
        from_stage="pairs", to_stage="evaluate",
        methods=[DENSE_FT_METHOD],
    )

    assert all(command.argv[2] == "--config" for command in commands)
    rendered = "\n".join(" ".join(command.argv) for command in commands)
    assert "--method" not in rendered
    assert "--model_dir" not in rendered
    assert "--checkpoint" not in rendered
    assert "--train_graphs" not in rendered
    assert "--encoder_model" not in rendered
