from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from graph_memory.registry.ids import StrEnum
from graph_memory.registry.retrieval import (
    Bm25RetrievalSettings,
    DenseFinetunedRetrievalSettings,
    DenseRetrievalSettings,
    EvidenceRgcnRetrievalSettings,
    ExecutionProvenanceRetrievalSettings,
    GraphRAGRetrievalSettings,
    RetrievalMethodId,
)
from graph_memory.registry.semantics import (
    MethodInputSpec,
    RequiredArtifact,
    RetrievalCapabilities,
    RetrievalTaskFamily,
)
from graph_memory.retrieval.requests import (
    EvidenceGraphRankingRequest,
    ExecutionProvenanceRankingRequest,
    GraphRAGRequest,
    TextRankingRequest,
)


class RetrievalLifecycle(StrEnum):
    STATELESS = "stateless"
    RGCN_TRAINABLE = "rgcn_trainable"
    DENSE_FINETUNE = "dense_finetune"


class SelectedConfigSource(StrEnum):
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
    selected_config: SelectedConfigSource
    model: ModelSource
    encoder: EncoderSource


@dataclass(frozen=True)
class MethodDefinition:
    identifier: RetrievalMethodId
    lifecycle: RetrievalLifecycle
    retrieval_settings_type: type[object]
    input_spec: MethodInputSpec
    capabilities: RetrievalCapabilities
    dependencies: RetrievalDependencySpec
    train_artifact: TrainArtifactSpec | None
    seed_method: RetrievalMethodId | None = None
    train_dependencies: tuple[RetrievalMethodId, ...] = ()


@dataclass(frozen=True)
class MethodRegistry:
    definitions: Mapping[RetrievalMethodId, MethodDefinition]

    def list_ids(self) -> tuple[RetrievalMethodId, ...]:
        return tuple(
            method for method in RetrievalMethodId if method in self.definitions
        )

    def get(self, method: str | RetrievalMethodId) -> MethodDefinition:
        try:
            method_id = (
                method
                if isinstance(method, RetrievalMethodId)
                else RetrievalMethodId(method)
            )
            return self.definitions[method_id]
        except (KeyError, ValueError) as error:
            raise ValueError(f"Unsupported retrieval method: {method}") from error

    def list_by_lifecycle(
        self, lifecycle: RetrievalLifecycle
    ) -> tuple[RetrievalMethodId, ...]:
        return tuple(
            method
            for method in self.list_ids()
            if self.definitions[method].lifecycle is lifecycle
        )

    def list_by_family(
        self, family: RetrievalTaskFamily
    ) -> tuple[RetrievalMethodId, ...]:
        return tuple(
            method
            for method in self.list_ids()
            if family in self.definitions[method].input_spec.supported_families
        )

    def requires_artifact(
        self,
        method: str | RetrievalMethodId,
        artifact: RequiredArtifact,
    ) -> bool:
        return self.get(method).input_spec.required_artifact is artifact

    def validate_request(
        self,
        method: str | RetrievalMethodId,
        request: object,
        family: RetrievalTaskFamily,
    ) -> None:
        definition = self.get(method)
        if not isinstance(request, definition.input_spec.request_type):
            raise TypeError(
                f"method={definition.identifier.value} requires "
                f"{definition.input_spec.request_type.__name__}, got {type(request).__name__}."
            )
        if family not in definition.input_spec.supported_families:
            supported = ", ".join(
                sorted(item.value for item in definition.input_spec.supported_families)
            )
            raise TypeError(
                f"method={definition.identifier.value} does not support family={family.value}; "
                f"supported families: {supported}."
            )

    def expand_train_dependencies(
        self,
        methods: tuple[RetrievalMethodId, ...],
    ) -> tuple[RetrievalMethodId, ...]:
        ordered: list[RetrievalMethodId] = []
        visited: set[RetrievalMethodId] = set()
        visiting: set[RetrievalMethodId] = set()

        def visit(method: RetrievalMethodId) -> None:
            if method in visited:
                return
            if method in visiting:
                raise ValueError(f"train dependency cycle at method={method.value}")
            visiting.add(method)
            definition = self.definitions[method]
            for dependency in definition.train_dependencies:
                visit(dependency)
            visiting.remove(method)
            visited.add(method)
            if definition.train_artifact is not None:
                ordered.append(method)

        for method in methods:
            visit(method)
        return tuple(ordered)


def build_method_registry() -> MethodRegistry:
    evidence = frozenset({RetrievalTaskFamily.EVIDENCE_RETRIEVAL})
    provenance = frozenset({RetrievalTaskFamily.EXECUTION_PROVENANCE})
    shared = evidence | provenance
    no_dependencies = RetrievalDependencySpec(
        selected_config=SelectedConfigSource.NONE,
        model=ModelSource.NONE,
        encoder=EncoderSource.NONE,
    )
    definitions = (
        MethodDefinition(
            identifier=RetrievalMethodId.BM25,
            lifecycle=RetrievalLifecycle.STATELESS,
            retrieval_settings_type=Bm25RetrievalSettings,
            input_spec=MethodInputSpec(
                TextRankingRequest, RequiredArtifact.NONE, shared
            ),
            capabilities=RetrievalCapabilities(True, False, False),
            dependencies=no_dependencies,
            train_artifact=None,
        ),
        MethodDefinition(
            identifier=RetrievalMethodId.DENSE,
            lifecycle=RetrievalLifecycle.STATELESS,
            retrieval_settings_type=DenseRetrievalSettings,
            input_spec=MethodInputSpec(
                TextRankingRequest, RequiredArtifact.NONE, shared
            ),
            capabilities=RetrievalCapabilities(True, False, False),
            dependencies=RetrievalDependencySpec(
                selected_config=SelectedConfigSource.NONE,
                model=ModelSource.NONE,
                encoder=EncoderSource.EXPERIMENT_CONFIG,
            ),
            train_artifact=None,
        ),
        MethodDefinition(
            identifier=RetrievalMethodId.DENSE_FT,
            lifecycle=RetrievalLifecycle.DENSE_FINETUNE,
            retrieval_settings_type=DenseFinetunedRetrievalSettings,
            input_spec=MethodInputSpec(
                TextRankingRequest, RequiredArtifact.NONE, evidence
            ),
            capabilities=RetrievalCapabilities(True, False, True),
            dependencies=RetrievalDependencySpec(
                selected_config=SelectedConfigSource.NONE,
                model=ModelSource.MODEL_DIRECTORY,
                encoder=EncoderSource.CHECKPOINT_METADATA,
            ),
            train_artifact=TrainArtifactSpec("best_model", ArtifactKind.DIRECTORY),
            seed_method=RetrievalMethodId.DENSE,
        ),
        MethodDefinition(
            identifier=RetrievalMethodId.GRAPHRAG,
            lifecycle=RetrievalLifecycle.STATELESS,
            retrieval_settings_type=GraphRAGRetrievalSettings,
            input_spec=MethodInputSpec(GraphRAGRequest, RequiredArtifact.NONE, shared),
            capabilities=RetrievalCapabilities(True, True, False),
            dependencies=RetrievalDependencySpec(
                selected_config=SelectedConfigSource.NONE,
                model=ModelSource.NONE,
                encoder=EncoderSource.EXPERIMENT_CONFIG,
            ),
            train_artifact=None,
        ),
        MethodDefinition(
            identifier=RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER,
            lifecycle=RetrievalLifecycle.RGCN_TRAINABLE,
            retrieval_settings_type=EvidenceRgcnRetrievalSettings,
            input_spec=MethodInputSpec(
                EvidenceGraphRankingRequest,
                RequiredArtifact.EVIDENCE_GRAPH,
                evidence,
            ),
            capabilities=RetrievalCapabilities(True, False, True),
            dependencies=RetrievalDependencySpec(
                selected_config=SelectedConfigSource.NONE,
                model=ModelSource.CHECKPOINT_FILE,
                encoder=EncoderSource.CHECKPOINT_METADATA,
            ),
            train_artifact=TrainArtifactSpec("best.pt", ArtifactKind.FILE),
            seed_method=RetrievalMethodId.DENSE,
        ),
        MethodDefinition(
            identifier=RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER,
            lifecycle=RetrievalLifecycle.RGCN_TRAINABLE,
            retrieval_settings_type=EvidenceRgcnRetrievalSettings,
            input_spec=MethodInputSpec(
                EvidenceGraphRankingRequest,
                RequiredArtifact.EVIDENCE_GRAPH,
                evidence,
            ),
            capabilities=RetrievalCapabilities(True, False, True),
            dependencies=RetrievalDependencySpec(
                selected_config=SelectedConfigSource.NONE,
                model=ModelSource.CHECKPOINT_FILE,
                encoder=EncoderSource.CHECKPOINT_METADATA,
            ),
            train_artifact=TrainArtifactSpec("best.pt", ArtifactKind.FILE),
            seed_method=RetrievalMethodId.DENSE_FT,
            train_dependencies=(RetrievalMethodId.DENSE_FT,),
        ),
        MethodDefinition(
            identifier=RetrievalMethodId.EXECUTION_PROVENANCE_RETRIEVER,
            lifecycle=RetrievalLifecycle.STATELESS,
            retrieval_settings_type=ExecutionProvenanceRetrievalSettings,
            input_spec=MethodInputSpec(
                ExecutionProvenanceRankingRequest,
                RequiredArtifact.NONE,
                provenance,
            ),
            capabilities=RetrievalCapabilities(True, True, False),
            dependencies=RetrievalDependencySpec(
                selected_config=SelectedConfigSource.NONE,
                model=ModelSource.NONE,
                encoder=EncoderSource.EXPERIMENT_CONFIG,
            ),
            train_artifact=None,
        ),
    )
    return MethodRegistry(
        {definition.identifier: definition for definition in definitions}
    )


__all__ = [
    "ArtifactKind",
    "EncoderSource",
    "MethodDefinition",
    "MethodInputSpec",
    "MethodRegistry",
    "ModelSource",
    "RequiredArtifact",
    "RetrievalCapabilities",
    "RetrievalDependencySpec",
    "RetrievalLifecycle",
    "RetrievalTaskFamily",
    "SelectedConfigSource",
    "TrainArtifactSpec",
    "build_method_registry",
]
