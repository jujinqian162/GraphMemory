from __future__ import annotations

from pathlib import Path

import pytest

from graph_memory.config import CONFIG_LOADER
from graph_memory.io import write_json
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
from graph_memory.registry.stage_configs import RetrieveIO, RetrieveStageConfig
from graph_memory.retrieval.execution.service import run_retrieval
from graph_memory.retrieval.methods.flat.dense import DenseTaskRetriever
from graph_memory.retrieval.methods.flat.method import ScorePipelineMethod
from graph_memory.stages.retrieve import run_retrieve_stage
from tests.test_phase1_real_retrieval import FakeEncoder, retrieval_task_inputs


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


def _stage_config(tmp_path: Path, checkpoint: Path) -> RetrieveStageConfig:
    return RetrieveStageConfig(
        io=RetrieveIO(
            tasks=tmp_path / "tasks.json",
            graphs=None,
            output=tmp_path / "predictions.json",
            summary=tmp_path / "predictions.run_summary.json",
        ),
        job=DenseFinetunedRetrievalSettings(
            top_k=2,
            checkpoint=checkpoint,
            device="cpu",
        ),
    )


def test_dense_ft_method_definition_requires_model_directory() -> None:
    definition = Registry.methods.get(RetrievalMethodId.DENSE_FT)

    assert definition.lifecycle is RetrievalLifecycle.DENSE_FINETUNE
    assert definition.dependencies.model is ModelSource.MODEL_DIRECTORY
    assert definition.train_artifact is not None
    assert definition.train_artifact.kind is ArtifactKind.DIRECTORY


def test_retrieve_stage_config_loads_dense_ft_from_complete_config(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best_model"
    config_path = tmp_path / "retrieve.json"
    expected = _stage_config(tmp_path, checkpoint)
    write_json(config_path, CONFIG_LOADER.to_json(expected))

    config = CONFIG_LOADER.load(Registry.configs.RETRIEVE, ["--config", str(config_path)])

    assert config == expected


def test_dense_ft_builder_loads_typed_metadata_and_reuses_dense_retriever(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best_model"
    _write_dense_ft_metadata(checkpoint)

    built = Registry.retrieval.build(
        DenseFinetunedRetrievalSettings(top_k=2, checkpoint=checkpoint, device="cpu"),
        FlatRetrievalBuildPayload(task_inputs=retrieval_task_inputs(), dense_encoder=FakeEncoder()),
    )

    assert isinstance(built.method, ScorePipelineMethod)
    assert isinstance(built.method.retriever, DenseTaskRetriever)
    assert built.method.retriever.config.model_name == str(checkpoint)
    assert built.method.retriever.config.query_prefix == "Q: "
    predictions = run_retrieval(
        retrieval_method=built.method,
        task_inputs=retrieval_task_inputs(),
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
            DenseFinetunedRetrievalSettings(top_k=2, checkpoint=checkpoint, device="cpu"),
            FlatRetrievalBuildPayload(task_inputs=retrieval_task_inputs(), dense_encoder=FakeEncoder()),
        )
