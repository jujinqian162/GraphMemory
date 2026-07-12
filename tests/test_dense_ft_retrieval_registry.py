from __future__ import annotations

from pathlib import Path

import pytest

from graph_memory.experiment.config import ArtifactRef
from graph_memory.experiment.invocation import StageInvocation
from graph_memory.experiment.persistence import write_yaml_atomic
from graph_memory.experiment.stage_cli import load_stage_execution
from graph_memory.experiment.stage_models import DenseFinetuneRetrieveStageConfig
from graph_memory.models.dense_finetune.metadata import (
    DenseFinetuneModelMetadata,
    DenseFinetuneSelectionMetadata,
    write_dense_ft_model_metadata,
)
from graph_memory.registry import Registry
from graph_memory.registry.methods import ArtifactKind, ModelSource, RetrievalLifecycle
from graph_memory.registry.retrieval import (
    DenseFinetunedRetrievalSettings,
    FlatRetrievalBuildPayload,
    RetrievalMethodId,
)
from graph_memory.retrieval.execution.service import run_retrieval
from graph_memory.retrieval.methods.flat.dense import DenseTaskRetriever
from graph_memory.retrieval.methods.flat.method import ScorePipelineMethod
from graph_memory.stages.retrieve import run_retrieve_stage
from tests.test_phase1_real_retrieval import (
    FakeEncoder,
    retrieval_ranking_requests,
    retrieval_task_inputs,
)


def _write_dense_ft_metadata(model_dir: Path) -> None:
    write_dense_ft_model_metadata(
        model_dir=model_dir,
        metadata=DenseFinetuneModelMetadata(
            base_model="fake-base",
            query_prefix="Q: ",
            passage_prefix="P: ",
            batch_size=7,
            device="cpu",
            selection=DenseFinetuneSelectionMetadata(
                selected_metric="eval_dev_cos_sim_map@100",
                higher_is_better=True,
            ),
        ),
    )


def _stage_config(tmp_path: Path, checkpoint: Path) -> DenseFinetuneRetrieveStageConfig:
    return DenseFinetuneRetrieveStageConfig(
        stage="retrieve",
        method="dense_ft",
        variant=None,
        dataset="hotpotqa",
        tasks=(tmp_path / "tasks.json").resolve(),
        output=(tmp_path / "predictions.json").resolve(),
        top_k=2,
        model_dir=checkpoint.resolve(),
        device="cpu",
    )


def test_dense_ft_method_definition_requires_model_directory() -> None:
    definition = Registry.methods.get(RetrievalMethodId.DENSE_FT)

    assert definition.lifecycle is RetrievalLifecycle.DENSE_FINETUNE
    assert definition.dependencies.model is ModelSource.MODEL_DIRECTORY
    assert definition.train_artifact is not None
    assert definition.train_artifact.kind is ArtifactKind.DIRECTORY


def test_retrieve_stage_config_loads_dense_ft_from_complete_config(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best_model"
    config_path = tmp_path / "retrieve.yaml"
    expected = _stage_config(tmp_path, checkpoint)
    summary = (tmp_path / "predictions.run_summary.yaml").resolve()
    script = (tmp_path / "run_retrieval.py").resolve()
    invocation = StageInvocation(
        identifier="retrieve:dense_ft",
        stage="retrieve",
        script=script,
        config_path=config_path.resolve(),
        summary_path=summary,
        config=expected,
        inputs=(
            ArtifactRef(
                role="inputs",
                path=expected.tasks,
                kind="file",
            ),
            ArtifactRef(
                role="checkpoint",
                path=expected.model_dir,
                kind="directory",
            ),
        ),
        outputs=(
            ArtifactRef(
                role="predictions",
                path=expected.output,
                kind="file",
            ),
        ),
        dependencies=("train:dense_ft",),
        method=RetrievalMethodId.DENSE_FT,
    )
    write_yaml_atomic(config_path, invocation)

    config = load_stage_execution(
        ["--config", str(config_path)],
        DenseFinetuneRetrieveStageConfig,
        description="test",
        script=script,
    ).config

    assert config == expected


def test_dense_ft_builder_loads_typed_metadata_and_reuses_dense_retriever(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best_model"
    _write_dense_ft_metadata(checkpoint)

    built = Registry.retrieval.build(
        DenseFinetunedRetrievalSettings(top_k=2, checkpoint=checkpoint, device="cpu"),
        FlatRetrievalBuildPayload(
            ranking_requests=retrieval_ranking_requests(), dense_encoder=FakeEncoder()
        ),
    )

    assert isinstance(built.method, ScorePipelineMethod)
    assert isinstance(built.method.retriever, DenseTaskRetriever)
    assert built.method.retriever.config.model_name == str(checkpoint)
    assert built.method.retriever.config.query_prefix == "Q: "
    predictions = run_retrieval(
        retrieval_method=built.method,
        tasks=built.execution_tasks,
        top_k=2,
    )
    assert predictions[0]["method"] == "dense_ft"

    result = run_retrieve_stage(
        _stage_config(tmp_path, checkpoint),
        task_inputs=retrieval_task_inputs(),
        graphs=None,
        dense_encoder=FakeEncoder(),
    )
    assert result.predictions[0]["method"] == "dense_ft"
    assert result.provenance.model == checkpoint


def test_dense_ft_builder_reports_missing_metadata_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "missing_model"
    checkpoint.mkdir()

    with pytest.raises(ValueError, match=r"dense_ft_model_config\.json.*missing_model"):
        Registry.retrieval.build(
            DenseFinetunedRetrievalSettings(
                top_k=2, checkpoint=checkpoint, device="cpu"
            ),
            FlatRetrievalBuildPayload(
                ranking_requests=retrieval_ranking_requests(),
                dense_encoder=FakeEncoder(),
            ),
        )
