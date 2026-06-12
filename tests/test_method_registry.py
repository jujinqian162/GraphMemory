from __future__ import annotations

import pytest

from graph_memory.registry import Registry
from graph_memory.registry.method_configs import DenseFinetuneMethodConfig, RgcnMethodConfig
from graph_memory.registry.methods import (
    ArtifactKind,
    EncoderSource,
    GraphConfigSource,
    GraphInputSource,
    ModelSource,
    RetrievalLifecycle,
)
from graph_memory.registry.retrieval import RetrievalMethodId


def test_registry_exposes_every_retrieval_method_once() -> None:
    assert Registry.methods.list_ids() == tuple(RetrievalMethodId)


def test_rgcn_method_definition_is_complete() -> None:
    definition = Registry.methods.get(RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER)

    assert definition.lifecycle is RetrievalLifecycle.RGCN_TRAINABLE
    assert definition.dependencies.graphs is GraphInputSource.GRAPH_ARTIFACT
    assert definition.dependencies.graph_config is GraphConfigSource.NONE
    assert definition.dependencies.model is ModelSource.CHECKPOINT_FILE
    assert definition.dependencies.encoder is EncoderSource.CHECKPOINT_METADATA
    assert definition.method_config_type is RgcnMethodConfig
    assert definition.train_artifact is not None
    assert definition.train_artifact.basename == "best.pt"
    assert definition.train_artifact.kind is ArtifactKind.FILE


def test_dense_ft_method_definition_declares_model_directory() -> None:
    definition = Registry.methods.get(RetrievalMethodId.DENSE_FT)

    assert definition.lifecycle is RetrievalLifecycle.DENSE_FINETUNE
    assert definition.dependencies.graphs is GraphInputSource.NONE
    assert definition.dependencies.model is ModelSource.MODEL_DIRECTORY
    assert definition.dependencies.encoder is EncoderSource.CHECKPOINT_METADATA
    assert definition.method_config_type is DenseFinetuneMethodConfig
    assert definition.train_artifact is not None
    assert definition.train_artifact.basename == "best_model"
    assert definition.train_artifact.kind is ArtifactKind.DIRECTORY


def test_registry_lists_graph_rerank_methods_by_lifecycle() -> None:
    assert Registry.methods.list_by_lifecycle(RetrievalLifecycle.GRAPH_RERANK) == (
        RetrievalMethodId.BM25_GRAPH_RERANK,
        RetrievalMethodId.DENSE_GRAPH_RERANK,
    )


def test_method_definitions_do_not_expose_capability_booleans_or_builder_ids() -> None:
    retired_attributes = {
        "builder_id",
        "requires_graphs",
        "requires_graph_config",
        "requires_checkpoint",
        "requires_dense_encoder",
    }

    for method in Registry.methods.list_ids():
        assert retired_attributes.isdisjoint(vars(Registry.methods.get(method)))


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported retrieval method"):
        Registry.methods.get("unknown")
