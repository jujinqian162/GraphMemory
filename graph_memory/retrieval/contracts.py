from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeAlias

from graph_memory.contracts.common import NodeId, Score
from graph_memory.contracts.graphs import GraphEdge
from graph_memory.graphs.provenance import ProvenanceEdgeType
from graph_memory.retrieval.requests import RankingMethodRequest, TextRankingRequest


@dataclass(frozen=True)
class RankedNode:
    node_id: NodeId
    score: Score


@dataclass(frozen=True)
class EntityRelationTrace:
    source_entity_id: str
    target_entity_id: str
    weight: float
    candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class GraphRAGTrace:
    entity_ids: tuple[str, ...]
    linked_entity_ids: tuple[str, ...]
    seed_entity_ids: tuple[str, ...]
    relations: tuple[EntityRelationTrace, ...]
    trace_kind: Literal["entity_search"] = "entity_search"


@dataclass(frozen=True)
class ProvenanceBindingTrace:
    output_field: str
    input_parameter: str
    binding_kind: str


@dataclass(frozen=True)
class ProvenanceEdgeTrace:
    source: str
    target: str
    edge_type: ProvenanceEdgeType
    weight: float
    binding: ProvenanceBindingTrace | None = None


@dataclass(frozen=True)
class ProvenancePathTrace:
    node_ids: tuple[str, ...]
    score: float
    semantic_relevance: float
    binding_consistency: float
    provenance_completeness: float
    explicit_grounding: float
    path_length_penalty: float
    invalidation_penalty: float


@dataclass(frozen=True)
class ExecutionProvenanceTrace:
    node_ids: tuple[str, ...]
    paths: tuple[ProvenancePathTrace, ...]
    edges: tuple[ProvenanceEdgeTrace, ...]
    trace_kind: Literal["execution_provenance"] = "execution_provenance"


NativeRetrievalTrace: TypeAlias = GraphRAGTrace | ExecutionProvenanceTrace


@dataclass(frozen=True)
class RetrievalTrace:
    retrieved_edges: list[GraphEdge] = field(default_factory=list)
    native_trace: NativeRetrievalTrace | None = None


@dataclass(frozen=True)
class RetrievalMethodResult:
    ranked_nodes: list[RankedNode]
    trace: RetrievalTrace = field(default_factory=RetrievalTrace)


class SeedRanker(Protocol):
    @property
    def method_name(self) -> str: ...

    def rank(self, request: TextRankingRequest) -> list[RankedNode]: ...


class RetrievalMethod(Protocol):
    @property
    def name(self) -> str: ...

    def rank_task(
        self,
        request: RankingMethodRequest,
        *,
        top_k: int,
    ) -> RetrievalMethodResult: ...
