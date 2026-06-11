from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from graph_memory.registry.retrieval import (
    Bm25RetrievalSettings,
    CheckpointGraphRetrievalSettings,
    DenseRetrievalSettings,
    GraphRerankRetrievalSettings,
    RETRIEVAL_METHOD_METADATA,
    RetrievalMethodMetadata,
)


@dataclass(frozen=True)
class RetrievalMethodSpec:
    """
    Compatibility metadata for one public retrieval method.

    `builder_id` is a legacy projection field. New runtime dispatch must use
    registry-owned settings types and builder specs instead.
    """

    name: str
    requires_graphs: bool
    requires_graph_config: bool
    requires_checkpoint: bool
    requires_dense_encoder: bool
    seed_method: str | None
    builder_id: str


def _legacy_builder_id_for(metadata: RetrievalMethodMetadata) -> str:
    if metadata.settings_type is Bm25RetrievalSettings:
        return "bm25"
    if metadata.settings_type is DenseRetrievalSettings:
        return "dense"
    if metadata.settings_type is GraphRerankRetrievalSettings:
        return "graph_rerank"
    if metadata.settings_type is CheckpointGraphRetrievalSettings:
        return "trainable_graph"
    raise ValueError(f"Unsupported retrieval settings projection: {metadata.settings_type.__name__}")


def _project_retrieval_method_registry(
    metadata: Mapping[str, RetrievalMethodMetadata] = RETRIEVAL_METHOD_METADATA,
) -> dict[str, RetrievalMethodSpec]:
    return {
        method: RetrievalMethodSpec(
            name=source.name,
            requires_graphs=source.requires_graphs,
            requires_graph_config=source.requires_graph_config,
            requires_checkpoint=source.requires_checkpoint,
            requires_dense_encoder=source.requires_dense_encoder,
            seed_method=source.seed_method.value if source.seed_method is not None else None,
            builder_id=_legacy_builder_id_for(source),
        )
        for method, source in metadata.items()
    }


METHOD_REGISTRY: dict[str, RetrievalMethodSpec] = _project_retrieval_method_registry()


def get_supported_methods() -> tuple[str, ...]:
    return tuple(METHOD_REGISTRY)


def get_graph_rerank_methods() -> tuple[str, ...]:
    return tuple(method for method, spec in METHOD_REGISTRY.items() if spec.builder_id == "graph_rerank")


def get_methods_requiring_dense_encoder() -> tuple[str, ...]:
    return tuple(method for method, spec in METHOD_REGISTRY.items() if spec.requires_dense_encoder)


def get_method_spec(method: str) -> RetrievalMethodSpec:
    try:
        return METHOD_REGISTRY[method]
    except KeyError as error:
        raise ValueError(f"Unsupported retrieval method: {method}") from error


__all__ = [
    "METHOD_REGISTRY",
    "RetrievalMethodSpec",
    "get_graph_rerank_methods",
    "get_method_spec",
    "get_methods_requiring_dense_encoder",
    "get_supported_methods",
]
