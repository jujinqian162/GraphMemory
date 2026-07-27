from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFiniteFloat = Annotated[float, Field(allow_inf_nan=False, ge=0.0)]


class _NativeTraceModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ProvenanceBindingTrace(_NativeTraceModel):
    output_field: str
    input_parameter: str
    binding_kind: str


class ProvenanceEdgeTrace(_NativeTraceModel):
    source: str
    target: str
    edge_type: ProvenanceEdgeType
    weight: NonNegativeFiniteFloat
    binding: ProvenanceBindingTrace | None = None
    semantic_rank: Annotated[int, Field(ge=1)] | None = None
    semantic_score: FiniteFloat | None = None


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
    variant: str = "typed_beam"
    trace_kind: Literal["execution_provenance_local"] = "execution_provenance_local"


@dataclass(frozen=True)
class ProvenanceRelationAffinityTrace:
    edge_type: str
    similarity: float
    affinity: float


@dataclass(frozen=True)
class ProvenanceTransitionTrace:
    source: str
    target: str
    edge_type: str
    direction: Literal["forward", "reverse"]
    recorded_weight: float
    relation_affinity: float
    probability: float
    cost: float


@dataclass(frozen=True)
class ProvenancePprNodeTrace:
    node_id: str
    teleport: float
    score: float


@dataclass(frozen=True)
class ProvenanceCandidatePrizeTrace:
    node_id: str
    dense_component: float
    ppr_component: float
    prize: float


@dataclass(frozen=True)
class ProvenanceSelectedArcTrace:
    source: str
    target: str
    edge_type: str
    direction: Literal["forward", "reverse"]


@dataclass(frozen=True)
class ProvenanceSelectionStepTrace:
    anchor_id: str
    target_id: str
    path_node_ids: tuple[str, ...]
    transitions: tuple[ProvenanceSelectedArcTrace, ...]
    added_candidate_ids: tuple[str, ...]
    displaced_candidate_ids: tuple[str, ...]
    prize_gain: float
    edge_cost: float
    displacement_cost: float
    marginal_gain: float


@dataclass(frozen=True)
class QueryConditionedExecutionProvenanceTrace:
    native_graph_node_ids: tuple[str, ...]
    dense_ranks: tuple[DenseRankTrace, ...]
    relation_description_version: str
    relations: tuple[ProvenanceRelationAffinityTrace, ...]
    transitions: tuple[ProvenanceTransitionTrace, ...]
    ppr_nodes: tuple[ProvenancePprNodeTrace, ...]
    ppr_iterations: int
    ppr_residual: float
    ppr_converged: bool
    candidate_prizes: tuple[ProvenanceCandidatePrizeTrace, ...]
    selected_candidate_ids: tuple[str, ...]
    connector_node_ids: tuple[str, ...]
    selection_steps: tuple[ProvenanceSelectionStepTrace, ...]
    selected_native_edges: tuple[ProvenanceEdgeTrace, ...]
    objective: float
    top_k: int
    exact_dense_fallback: bool
    emitted_edges: tuple[CandidateEdgeTrace, ...]
    scorer_identity: str
    variant: str = "ppr_steiner"
    trace_kind: Literal["execution_provenance_subgraph"] = (
        "execution_provenance_subgraph"
    )


class ProvenancePathTrace(_NativeTraceModel):
    node_ids: tuple[str, ...] = Field(min_length=1)
    score: FiniteFloat
    semantic_relevance: FiniteFloat
    binding_consistency: FiniteFloat
    provenance_completeness: FiniteFloat
    explicit_grounding: FiniteFloat
    path_length_penalty: FiniteFloat
    invalidation_penalty: FiniteFloat


class ProvenanceStructuredTransitionTrace(_NativeTraceModel):
    source_id: str
    target_id: str
    probability: NonNegativeFiniteFloat
    original_source_rank: Annotated[int, Field(ge=0)]
    original_target_rank: Annotated[int, Field(ge=0)]
    final_target_rank: Annotated[int, Field(ge=0)]
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


class ExecutionProvenanceTrace(_NativeTraceModel):
    node_ids: tuple[str, ...]
    paths: tuple[ProvenancePathTrace, ...]
    edges: tuple[ProvenanceEdgeTrace, ...]
    structured_transitions: tuple[ProvenanceStructuredTransitionTrace, ...] = ()
    abstained_source_ids: tuple[str, ...] = ()
    structured_promotion_enabled: bool = True
    trace_kind: Literal["execution_provenance"] = "execution_provenance"

    @model_validator(mode="after")
    def _check_graph_invariants(self) -> "ExecutionProvenanceTrace":
        node_id_set = set(self.node_ids)
        if len(self.node_ids) != len(node_id_set):
            raise ValueError("node_ids contains duplicates")

        seen_edges: set[tuple[str, str, str]] = set()
        adjacent_pairs: set[frozenset[str]] = set()
        for edge in self.edges:
            if edge.source not in node_id_set or edge.target not in node_id_set:
                raise ValueError(
                    f"edge endpoint {edge.source}->{edge.target} is unknown"
                )
            if edge.source == edge.target:
                raise ValueError("edge cannot be a self loop")
            edge_key = (edge.source, edge.target, edge.edge_type.value)
            if edge_key in seen_edges:
                raise ValueError(f"duplicate edge={edge_key}")
            seen_edges.add(edge_key)
            adjacent_pairs.add(frozenset((edge.source, edge.target)))
            if edge.edge_type is ProvenanceEdgeType.FEEDS:
                if edge.binding is None:
                    raise ValueError("feeds edge requires a binding")
            elif edge.binding is not None:
                raise ValueError("binding is only valid on feeds edges")

        seen_paths: set[tuple[str, ...]] = set()
        for path in self.paths:
            if path.node_ids in seen_paths:
                raise ValueError(f"duplicate path={list(path.node_ids)}")
            seen_paths.add(path.node_ids)
            if len(path.node_ids) != len(set(path.node_ids)):
                raise ValueError("path.node_ids contains duplicates")
            unknown_nodes = sorted(set(path.node_ids) - node_id_set)
            if unknown_nodes:
                raise ValueError(
                    f"path references unknown nodes={unknown_nodes}"
                )
            for source, target in zip(
                path.node_ids, path.node_ids[1:], strict=False
            ):
                if frozenset((source, target)) not in adjacent_pairs:
                    raise ValueError(
                        f"path step {source}->{target} has no traced edge"
                    )

        if len(self.abstained_source_ids) != len(set(self.abstained_source_ids)):
            raise ValueError("abstained_source_ids contains duplicates")

        seen_transitions: set[tuple[str, str]] = set()
        for transition in self.structured_transitions:
            transition_key = (transition.source_id, transition.target_id)
            if transition_key in seen_transitions:
                raise ValueError(
                    f"duplicate structured_transition={transition.source_id}->{transition.target_id}"
                )
            seen_transitions.add(transition_key)
        return self


NativeRetrievalTrace: TypeAlias = (
    GraphRAGTrace
    | ExecutionProvenanceTrace
    | StatelessExecutionProvenanceTrace
    | QueryConditionedExecutionProvenanceTrace
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
