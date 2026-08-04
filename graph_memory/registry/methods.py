from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from graph_memory.registry.retrieval import RetrievalMethodId, RetrievalTaskFamily
from graph_memory.retrieval.requests import (
    EvidenceGraphRankingRequest,
    GraphRAGRequest,
    ProvenancePathRequest,
    ProvenanceRgcnRequest,
    TextRankingRequest,
)


@dataclass(frozen=True)
class MethodDefinition:
    identifier: RetrievalMethodId
    request_type: type[object]
    supported_families: frozenset[RetrievalTaskFamily]


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

    def validate_request(
        self,
        method: str | RetrievalMethodId,
        request: object,
        family: RetrievalTaskFamily,
    ) -> None:
        definition = self.get(method)
        if not isinstance(request, definition.request_type):
            raise TypeError(
                f"method={definition.identifier.value} requires "
                f"{definition.request_type.__name__}, got {type(request).__name__}."
            )
        if family not in definition.supported_families:
            supported = ", ".join(
                sorted(item.value for item in definition.supported_families)
            )
            raise TypeError(
                f"method={definition.identifier.value} does not support "
                f"family={family.value}; supported families: {supported}."
            )


def build_method_registry() -> MethodRegistry:
    evidence = frozenset({RetrievalTaskFamily.EVIDENCE_RETRIEVAL})
    provenance = frozenset({RetrievalTaskFamily.EXECUTION_PROVENANCE})
    shared = evidence | provenance
    definitions = (
        MethodDefinition(
            RetrievalMethodId.BM25,
            TextRankingRequest,
            shared,
        ),
        MethodDefinition(
            RetrievalMethodId.DENSE,
            TextRankingRequest,
            shared,
        ),
        MethodDefinition(
            RetrievalMethodId.DENSE_FT,
            TextRankingRequest,
            shared,
        ),
        MethodDefinition(
            RetrievalMethodId.GRAPHRAG,
            GraphRAGRequest,
            shared,
        ),
        MethodDefinition(
            RetrievalMethodId.PROVENANCE_PATH,
            ProvenancePathRequest,
            provenance,
        ),
        MethodDefinition(
            RetrievalMethodId.PROVENANCE_RGCN,
            ProvenanceRgcnRequest,
            provenance,
        ),
        MethodDefinition(
            RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER,
            EvidenceGraphRankingRequest,
            evidence,
        ),
        MethodDefinition(
            RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER,
            EvidenceGraphRankingRequest,
            evidence,
        ),
    )
    return MethodRegistry(
        {definition.identifier: definition for definition in definitions}
    )


__all__ = [
    "MethodDefinition",
    "MethodRegistry",
    "RetrievalTaskFamily",
    "build_method_registry",
]
