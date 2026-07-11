from __future__ import annotations

from dataclasses import dataclass

from graph_memory.registry import Registry
from graph_memory.registry.methods import (
    ArtifactKind,
    EncoderSource,
    GraphInputSource,
    ModelSource,
    RetrievalLifecycle,
    SelectedConfigSource,
    TuningKind,
)
from graph_memory.registry.retrieval import RetrievalMethodId


@dataclass(frozen=True)
class ExperimentMethodSpec:
    method: RetrievalMethodId
    lifecycle: RetrievalLifecycle
    graphs: GraphInputSource
    selected_config: SelectedConfigSource
    model: ModelSource
    encoder: EncoderSource
    tuning: TuningKind | None
    train_artifact_kind: ArtifactKind | None
    seed_method: RetrievalMethodId | None
    train_dependencies: tuple[RetrievalMethodId, ...]

    @property
    def trainable(self) -> bool:
        return self.train_artifact_kind is not None


class ExperimentMethodRegistry:
    def __init__(self, specs: tuple[ExperimentMethodSpec, ...]) -> None:
        self._specs = {spec.method: spec for spec in specs}
        expected = tuple(Registry.methods.list_ids())
        if tuple(self._specs) != expected:
            raise ValueError(
                f"experiment registry order mismatch: expected={expected}, actual={tuple(self._specs)}"
            )

    def list_ids(self) -> tuple[RetrievalMethodId, ...]:
        return tuple(self._specs)

    def get(self, method: str | RetrievalMethodId) -> ExperimentMethodSpec:
        try:
            method_id = (
                method
                if isinstance(method, RetrievalMethodId)
                else RetrievalMethodId(method)
            )
            return self._specs[method_id]
        except (KeyError, ValueError) as error:
            raise ValueError(f"Unsupported experiment method: {method}") from error

    def expand_train_dependencies(
        self,
        methods: tuple[RetrievalMethodId, ...] | list[RetrievalMethodId],
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
            spec = self.get(method)
            for dependency in spec.train_dependencies:
                visit(dependency)
            visiting.remove(method)
            visited.add(method)
            if spec.trainable:
                ordered.append(method)

        for method in methods:
            visit(method)
        return tuple(ordered)


def build_experiment_method_registry() -> ExperimentMethodRegistry:
    return ExperimentMethodRegistry(
        tuple(
            ExperimentMethodSpec(
                method=method,
                lifecycle=definition.lifecycle,
                graphs=definition.dependencies.graphs,
                selected_config=definition.dependencies.selected_config,
                model=definition.dependencies.model,
                encoder=definition.dependencies.encoder,
                tuning=definition.tuning,
                train_artifact_kind=(
                    definition.train_artifact.kind
                    if definition.train_artifact is not None
                    else None
                ),
                seed_method=definition.seed_method,
                train_dependencies=definition.train_dependencies,
            )
            for method in Registry.methods.list_ids()
            for definition in (Registry.methods.get(method),)
        )
    )


METHODS = build_experiment_method_registry()

__all__ = [
    "ExperimentMethodRegistry",
    "ExperimentMethodSpec",
    "METHODS",
    "build_experiment_method_registry",
]
