from __future__ import annotations

import json
from pathlib import Path

import pytest

from graph_memory.config import CONFIG_LOADER
from graph_memory.registry import Registry
from graph_memory.registry.projections import get_method_spec
from graph_memory.registry.retrieval import DenseFinetunedRetrievalSettings, FlatRetrievalBuildPayload, RetrievalMethodId
from graph_memory.retrieval.execution.service import run_retrieval
from graph_memory.retrieval.methods.flat.dense import DenseTaskRetriever
from graph_memory.retrieval.methods.flat.method import ScorePipelineMethod
from graph_memory.stages.retrieve import run_retrieve_stage
from tests.test_phase1_real_retrieval import FakeEncoder, retrieval_task_inputs


def _write_dense_ft_metadata(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "dense_ft_model_config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": "dense_ft",
                "base_model": "fake-base",
                "query_prefix": "Q: ",
                "passage_prefix": "P: ",
                "batch_size": 7,
                "selection": {
                    "selected_metric": "eval_dev_cos_sim_map@100",
                    "higher_is_better": True,
                },
            }
        ),
        encoding="utf-8",
    )


def test_dense_ft_retrieval_metadata_requires_model_directory_checkpoint() -> None:
    metadata = Registry.retrieval.metadata[RetrievalMethodId.DENSE_FT.value]
    projected = get_method_spec(RetrievalMethodId.DENSE_FT.value)

    assert metadata.name == "dense_ft"
    assert metadata.settings_type is DenseFinetunedRetrievalSettings
    assert metadata.requires_graphs is False
    assert metadata.requires_graph_config is False
    assert metadata.requires_checkpoint is True
    assert metadata.requires_dense_encoder is True
    assert metadata.seed_method is RetrievalMethodId.DENSE
    assert projected.builder_id == "dense"


def test_retrieve_stage_config_parses_dense_ft_checkpoint_without_graphs(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best_model"

    config = CONFIG_LOADER.load(
        Registry.configs.RETRIEVE,
        [
            "--method",
            "dense_ft",
            "--tasks",
            str(tmp_path / "tasks.json"),
            "--output",
            str(tmp_path / "predictions.json"),
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
        ],
    )

    assert config.io.graphs is None
    assert config.job == DenseFinetunedRetrievalSettings(top_k=10, checkpoint=checkpoint, device="cpu")


def test_dense_ft_builder_loads_metadata_and_reuses_dense_task_retriever(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best_model"
    _write_dense_ft_metadata(checkpoint)

    method = Registry.retrieval.build(
        DenseFinetunedRetrievalSettings(top_k=2, checkpoint=checkpoint, device="cpu"),
        FlatRetrievalBuildPayload(task_inputs=retrieval_task_inputs(), dense_encoder=FakeEncoder()),
    )

    assert isinstance(method, ScorePipelineMethod)
    assert method.name == "dense_ft"
    assert isinstance(method.retriever, DenseTaskRetriever)
    assert method.retriever.config.model_name == str(checkpoint)
    assert method.retriever.config.query_prefix == "Q: "
    assert method.retriever.config.passage_prefix == "P: "
    assert method.retriever.config.batch_size == 7
    predictions = run_retrieval(retrieval_method=method, task_inputs=retrieval_task_inputs(), top_k=2)
    assert predictions[0]["method"] == "dense_ft"


def test_dense_ft_retrieve_stage_uses_flat_payload(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best_model"
    _write_dense_ft_metadata(checkpoint)
    config = CONFIG_LOADER.load(
        Registry.configs.RETRIEVE,
        [
            "--method",
            "dense_ft",
            "--tasks",
            str(tmp_path / "tasks.json"),
            "--output",
            str(tmp_path / "predictions.json"),
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
        ],
    )

    result = run_retrieve_stage(
        config,
        task_inputs=retrieval_task_inputs(),
        graphs=None,
        dense_encoder=FakeEncoder(),
    )

    assert result.predictions[0]["method"] == "dense_ft"


def test_dense_ft_builder_reports_missing_metadata_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "missing_model"
    checkpoint.mkdir()

    with pytest.raises(ValueError, match=r"dense_ft_model_config\.json.*missing_model"):
        Registry.retrieval.build(
            DenseFinetunedRetrievalSettings(top_k=2, checkpoint=checkpoint, device="cpu"),
            FlatRetrievalBuildPayload(task_inputs=retrieval_task_inputs(), dense_encoder=FakeEncoder()),
        )
