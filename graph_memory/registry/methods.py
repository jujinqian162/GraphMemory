from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from graph_memory.registry.ids import StrEnum
from graph_memory.registry.method_configs import DenseFinetuneMethodConfig, RgcnMethodConfig
from graph_memory.registry.retrieval import (
    Bm25RetrievalSettings,
    CheckpointGraphRetrievalSettings,
    DenseFinetunedRetrievalSettings,
    DenseRetrievalSettings,
    GraphRerankRetrievalSettings,
    MemoryStreamRetrievalSettings,
    RetrievalMethodId,
)


class RetrievalLifecycle(StrEnum):
    STATELESS = "stateless"
    GRAPH_RERANK = "graph_rerank"
    RGCN_TRAINABLE = "rgcn_trainable"
    DENSE_FINETUNE = "dense_finetune"


class GraphInputSource(StrEnum):
    NONE = "none"
    GRAPH_ARTIFACT = "graph_artifact"


class GraphConfigSource(StrEnum):
    NONE = "none"
    TUNED_ARTIFACT = "tuned_artifact"


class ModelSource(StrEnum):
    NONE = "none"
    CHECKPOINT_FILE = "checkpoint_file"
    MODEL_DIRECTORY = "model_directory"


class EncoderSource(StrEnum):
    NONE = "none"
    EXPERIMENT_CONFIG = "experiment_config"
    CHECKPOINT_METADATA = "checkpoint_metadata"


class ArtifactKind(StrEnum):
    FILE = "file"
    DIRECTORY = "directory"


@dataclass(frozen=True)
class TrainArtifactSpec:
    basename: str
    kind: ArtifactKind


@dataclass(frozen=True)
class RetrievalDependencySpec:
    graphs: GraphInputSource
    graph_config: GraphConfigSource
    model: ModelSource
    encoder: EncoderSource


@dataclass(frozen=True)
class MethodDefinition:
    identifier: RetrievalMethodId
    lifecycle: RetrievalLifecycle
    retrieval_settings_type: type[object]
    dependencies: RetrievalDependencySpec
    method_config_type: type[object] | None
    train_artifact: TrainArtifactSpec | None
    seed_method: RetrievalMethodId | None = None


@dataclass(frozen=True)
class MethodRegistry:
    definitions: Mapping[RetrievalMethodId, MethodDefinition]

    def list_ids(self) -> tuple[RetrievalMethodId, ...]:
        return tuple(method for method in RetrievalMethodId if method in self.definitions)

    def get(self, method: str | RetrievalMethodId) -> MethodDefinition:
        try:
            method_id = method if isinstance(method, RetrievalMethodId) else RetrievalMethodId(method)
            return self.definitions[method_id]
        except (KeyError, ValueError) as error:
            raise ValueError(f"Unsupported retrieval method: {method}") from error

    def list_by_lifecycle(self, lifecycle: RetrievalLifecycle) -> tuple[RetrievalMethodId, ...]:
        return tuple(
            method
            for method in self.list_ids()
            if self.definitions[method].lifecycle is lifecycle
        )


def build_method_registry() -> MethodRegistry:
    no_dependencies = RetrievalDependencySpec(
        graphs=GraphInputSource.NONE,
        graph_config=GraphConfigSource.NONE,
        model=ModelSource.NONE,
        encoder=EncoderSource.NONE,
    )
    definitions = (
        MethodDefinition(
            identifier=RetrievalMethodId.BM25,
            lifecycle=RetrievalLifecycle.STATELESS,
            retrieval_settings_type=Bm25RetrievalSettings,
            dependencies=no_dependencies,
            method_config_type=None,
            train_artifact=None,
        ),
        MethodDefinition(
            identifier=RetrievalMethodId.DENSE,
            lifecycle=RetrievalLifecycle.STATELESS,
            retrieval_settings_type=DenseRetrievalSettings,
            dependencies=RetrievalDependencySpec(
                graphs=GraphInputSource.NONE,
                graph_config=GraphConfigSource.NONE,
                model=ModelSource.NONE,
                encoder=EncoderSource.EXPERIMENT_CONFIG,
            ),
            method_config_type=None,
            train_artifact=None,
        ),
        MethodDefinition(
            identifier=RetrievalMethodId.MEMORY_STREAM,
            lifecycle=RetrievalLifecycle.STATELESS,
            retrieval_settings_type=MemoryStreamRetrievalSettings,
            dependencies=RetrievalDependencySpec(
                graphs=GraphInputSource.NONE,
                graph_config=GraphConfigSource.NONE,
                model=ModelSource.NONE,
                encoder=EncoderSource.EXPERIMENT_CONFIG,
            ),
            method_config_type=None,
            train_artifact=None,
            seed_method=RetrievalMethodId.DENSE,
        ),
        MethodDefinition(
            identifier=RetrievalMethodId.BM25_GRAPH_RERANK,
            lifecycle=RetrievalLifecycle.GRAPH_RERANK,
            retrieval_settings_type=GraphRerankRetrievalSettings,
            dependencies=RetrievalDependencySpec(
                graphs=GraphInputSource.GRAPH_ARTIFACT,
                graph_config=GraphConfigSource.TUNED_ARTIFACT,
                model=ModelSource.NONE,
                encoder=EncoderSource.NONE,
            ),
            method_config_type=None,
            train_artifact=None,
            seed_method=RetrievalMethodId.BM25,
        ),
        MethodDefinition(
            identifier=RetrievalMethodId.DENSE_GRAPH_RERANK,
            lifecycle=RetrievalLifecycle.GRAPH_RERANK,
            retrieval_settings_type=GraphRerankRetrievalSettings,
            dependencies=RetrievalDependencySpec(
                graphs=GraphInputSource.GRAPH_ARTIFACT,
                graph_config=GraphConfigSource.TUNED_ARTIFACT,
                model=ModelSource.NONE,
                encoder=EncoderSource.EXPERIMENT_CONFIG,
            ),
            method_config_type=None,
            train_artifact=None,
            seed_method=RetrievalMethodId.DENSE,
        ),
        MethodDefinition(
            identifier=RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER,
            lifecycle=RetrievalLifecycle.RGCN_TRAINABLE,
            retrieval_settings_type=CheckpointGraphRetrievalSettings,
            dependencies=RetrievalDependencySpec(
                graphs=GraphInputSource.GRAPH_ARTIFACT,
                graph_config=GraphConfigSource.NONE,
                model=ModelSource.CHECKPOINT_FILE,
                encoder=EncoderSource.CHECKPOINT_METADATA,
            ),
            method_config_type=RgcnMethodConfig,
            train_artifact=TrainArtifactSpec("best.pt", ArtifactKind.FILE),
            seed_method=RetrievalMethodId.DENSE,
        ),
        MethodDefinition(
            identifier=RetrievalMethodId.DENSE_FT,
            lifecycle=RetrievalLifecycle.DENSE_FINETUNE,
            retrieval_settings_type=DenseFinetunedRetrievalSettings,
            dependencies=RetrievalDependencySpec(
                graphs=GraphInputSource.NONE,
                graph_config=GraphConfigSource.NONE,
                model=ModelSource.MODEL_DIRECTORY,
                encoder=EncoderSource.CHECKPOINT_METADATA,
            ),
            method_config_type=DenseFinetuneMethodConfig,
            train_artifact=TrainArtifactSpec("best_model", ArtifactKind.DIRECTORY),
            seed_method=RetrievalMethodId.DENSE,
        ),
    )
    return MethodRegistry({definition.identifier: definition for definition in definitions})


__all__ = [
    "ArtifactKind",
    "EncoderSource",
    "GraphConfigSource",
    "GraphInputSource",
    "MethodDefinition",
    "MethodRegistry",
    "ModelSource",
    "RetrievalDependencySpec",
    "RetrievalLifecycle",
    "TrainArtifactSpec",
    "build_method_registry",
]
