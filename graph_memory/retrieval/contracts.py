from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeAlias

from graph_memory.contracts.common import NodeId, Score
from graph_memory.contracts.graphs import GraphEdge
from graph_memory.graphs.provenance import ProvenanceEdgeType
from graph_memory.retrieval.requests import (
    GraphRAGCandidateBridge,
    GraphRAGEntityMention,
    GraphRAGResolverEvidence,
    GraphRAGTitleEntityGroup,
    RankingMethodRequest,
    TextRankingRequest,
)


@dataclass(frozen=True)
class RankedNode:
    node_id: NodeId
    score: Score


@dataclass(frozen=True)
class DenseRankTrace:
    node_id: str
    dense_rank: int
    dense_score: float
    final_rank: int


@dataclass(frozen=True)
class CandidateEdgeTrace:
    source: str
    target: str
    edge_type: str
    confidence: float


@dataclass(frozen=True)
class GraphRAGBridgeTrace:
    bridge: GraphRAGCandidateBridge
    accepted: bool
    rejection_reason: str | None
    original_partner_rank: int
    final_partner_rank: int


@dataclass(frozen=True)
class GraphRAGTrace:
    dense_ranks: tuple[DenseRankTrace, ...]
    seed_candidate_ids: tuple[str, ...]
    linked_entity_ids: tuple[str, ...]
    mentions: tuple[GraphRAGEntityMention, ...]
    title_groups: tuple[GraphRAGTitleEntityGroup, ...]
    resolver_evidence: tuple[GraphRAGResolverEvidence, ...]
    bridges: tuple[GraphRAGBridgeTrace, ...]
    protected_prefix: tuple[str, ...]
    exact_dense_fallback: bool
    emitted_edges: tuple[CandidateEdgeTrace, ...]
    trace_kind: Literal["typed_local_bridge"] = "typed_local_bridge"


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
    semantic_rank: int | None = None
    semantic_score: float | None = None


@dataclass(frozen=True)
class StatelessProvenancePathTrace:
    anchor_id: str
    partner_id: str
    node_ids: tuple[str, ...]
    path_confidence: float
    binding_valid: bool
    completeness_valid: bool
    lifecycle_valid: bool
    accepted: bool
    rejection_reason: str | None
    original_partner_rank: int
    final_partner_rank: int


@dataclass(frozen=True)
class StatelessExecutionProvenanceTrace:
    dense_ranks: tuple[DenseRankTrace, ...]
    seed_candidate_ids: tuple[str, ...]
    paths: tuple[StatelessProvenancePathTrace, ...]
    edges: tuple[ProvenanceEdgeTrace, ...]
    protected_prefix: tuple[str, ...]
    exact_dense_fallback: bool
    emitted_edges: tuple[CandidateEdgeTrace, ...]
    scorer_identity: str
    trace_kind: Literal["execution_provenance_local"] = "execution_provenance_local"


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
class ProvenanceStructuredTransitionTrace:
    source_id: str
    target_id: str
    probability: float
    original_source_rank: int
    original_target_rank: int
    final_target_rank: int
    decision: Literal[
        "promoted",
        "already_above_source",
        "target_conflict",
        "lower_scoring_successor",
        "below_threshold",
        "outside_pool",
        "promotion_disabled",
        "stable_no_op",
    ]


@dataclass(frozen=True)
class ExecutionProvenanceTrace:
    node_ids: tuple[str, ...]
    paths: tuple[ProvenancePathTrace, ...]
    edges: tuple[ProvenanceEdgeTrace, ...]
    structured_transitions: tuple[ProvenanceStructuredTransitionTrace, ...] = ()
    abstained_source_ids: tuple[str, ...] = ()
    structured_promotion_enabled: bool = True
    trace_kind: Literal["execution_provenance"] = "execution_provenance"


NativeRetrievalTrace: TypeAlias = (
    GraphRAGTrace | ExecutionProvenanceTrace | StatelessExecutionProvenanceTrace
)


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
