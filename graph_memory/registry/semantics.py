from __future__ import annotations

from dataclasses import dataclass

from graph_memory.registry.ids import StrEnum


class RequiredArtifact(StrEnum):
    NONE = "none"
    EVIDENCE_GRAPH = "evidence_graph"


class RetrievalTaskFamily(StrEnum):
    EVIDENCE_RETRIEVAL = "evidence_retrieval"
    EXECUTION_PROVENANCE = "execution_provenance"


@dataclass(frozen=True)
class MethodInputSpec:
    request_type: type[object]
    required_artifact: RequiredArtifact
    supported_families: frozenset[RetrievalTaskFamily]


@dataclass(frozen=True)
class RetrievalCapabilities:
    produces_ranked_nodes: bool
    produces_native_edge_trace: bool
    trainable: bool


__all__ = [
    "MethodInputSpec",
    "RequiredArtifact",
    "RetrievalCapabilities",
    "RetrievalTaskFamily",
]
