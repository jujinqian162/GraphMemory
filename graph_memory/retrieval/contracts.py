from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Sequence
from typing import Annotated, Literal, Protocol, TypeAlias

from pydantic import Field, StrictBool, model_validator

from graph_memory.contracts.model import (
    DomainModel,
    FiniteFloat,
    NonEmptyStr,
    NonNegativeFiniteFloat,
    NonNegativeInt,
    PositiveFiniteFloat,
    PositiveInt,
)
from graph_memory.graphs.contracts import GraphEdge
from graph_memory.graphs.provenance import ProvenanceEdgeType
from graph_memory.retrieval.requests import (
    GraphRAGCandidateBridge,
    GraphRAGEntityMention,
    GraphRAGResolverEvidence,
    GraphRAGTitleEntityGroup,
    RankingMethodRequest,
    TextRankingRequest,
)


class RankedNode(DomainModel):
    node_id: NonEmptyStr
    score: FiniteFloat


class DenseRankTrace(DomainModel):
    node_id: NonEmptyStr
    dense_rank: PositiveInt
    dense_score: FiniteFloat
    final_rank: PositiveInt


class CandidateEdgeTrace(DomainModel):
    source: NonEmptyStr
    target: NonEmptyStr
    edge_type: NonEmptyStr
    confidence: NonNegativeFiniteFloat

    @model_validator(mode="after")
    def _reject_self_loop(self) -> "CandidateEdgeTrace":
        if self.source == self.target:
            raise ValueError("candidate edge cannot be a self loop")
        return self


class GraphRAGBridgeTrace(DomainModel):
    bridge: GraphRAGCandidateBridge
    accepted: StrictBool
    rejection_reason: NonEmptyStr | None
    original_partner_rank: PositiveInt
    final_partner_rank: PositiveInt

    @model_validator(mode="after")
    def _validate_outcome(self) -> "GraphRAGBridgeTrace":
        if self.accepted and self.rejection_reason is not None:
            raise ValueError("accepted bridge cannot have a rejection reason")
        if not self.accepted and self.rejection_reason is None:
            raise ValueError("rejected bridge requires a reason")
        return self


class _NativeTraceModel(DomainModel):
    def validate_candidate_context(self, valid_candidate_ids: frozenset[str]) -> None:
        del valid_candidate_ids


class EntityRelationTrace(DomainModel):
    source_entity_id: NonEmptyStr
    target_entity_id: NonEmptyStr
    weight: PositiveFiniteFloat
    candidate_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_relation(self) -> "EntityRelationTrace":
        if self.source_entity_id == self.target_entity_id:
            raise ValueError("entity relation cannot be a self loop")
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("entity relation candidate IDs must be unique")
        return self


class EntitySearchTrace(_NativeTraceModel):
    entity_ids: tuple[NonEmptyStr, ...]
    linked_entity_ids: tuple[NonEmptyStr, ...]
    seed_entity_ids: tuple[NonEmptyStr, ...]
    relations: tuple[EntityRelationTrace, ...]
    trace_kind: Literal["entity_search"] = "entity_search"

    @model_validator(mode="after")
    def _validate_entities(self) -> "EntitySearchTrace":
        if len(self.entity_ids) != len(set(self.entity_ids)):
            raise ValueError("entity_ids contains duplicates")
        entities = set(self.entity_ids)
        for name, values in (
            ("linked_entity_ids", self.linked_entity_ids),
            ("seed_entity_ids", self.seed_entity_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} contains duplicates")
            unknown = sorted(set(values) - entities)
            if unknown:
                raise ValueError(f"{name} references unknown entities={unknown}")
        seen: set[tuple[str, str]] = set()
        for relation in self.relations:
            if (
                relation.source_entity_id not in entities
                or relation.target_entity_id not in entities
            ):
                raise ValueError("entity relation endpoint is unknown")
            key = (
                min(relation.source_entity_id, relation.target_entity_id),
                max(relation.source_entity_id, relation.target_entity_id),
            )
            if key in seen:
                raise ValueError(f"duplicate entity relation={key}")
            seen.add(key)
        return self

    def validate_candidate_context(self, valid_candidate_ids: frozenset[str]) -> None:
        referenced = {
            candidate_id
            for relation in self.relations
            for candidate_id in relation.candidate_ids
        }
        _require_candidate_subset(referenced, valid_candidate_ids, "entity relations")


class GraphRAGTrace(_NativeTraceModel):
    dense_ranks: tuple[DenseRankTrace, ...]
    seed_candidate_ids: tuple[NonEmptyStr, ...]
    linked_entity_ids: tuple[NonEmptyStr, ...]
    mentions: tuple[GraphRAGEntityMention, ...]
    title_groups: tuple[GraphRAGTitleEntityGroup, ...]
    resolver_evidence: tuple[GraphRAGResolverEvidence, ...]
    bridges: tuple[GraphRAGBridgeTrace, ...]
    protected_prefix: tuple[NonEmptyStr, ...]
    exact_dense_fallback: StrictBool
    emitted_edges: tuple[CandidateEdgeTrace, ...]
    trace_kind: Literal["typed_local_bridge"] = "typed_local_bridge"

    @model_validator(mode="after")
    def _validate_trace(self) -> "GraphRAGTrace":
        _require_unique_dense_ranks(self.dense_ranks)
        _require_unique(self.seed_candidate_ids, "seed_candidate_ids")
        _require_unique(self.protected_prefix, "protected_prefix")
        _require_unique(self.linked_entity_ids, "linked_entity_ids")
        accepted_count = sum(bridge.accepted for bridge in self.bridges)
        if self.exact_dense_fallback != (accepted_count == 0):
            raise ValueError("exact fallback state is inconsistent")
        return self

    def validate_candidate_context(self, valid_candidate_ids: frozenset[str]) -> None:
        referenced = {
            *(rank.node_id for rank in self.dense_ranks),
            *self.seed_candidate_ids,
            *self.protected_prefix,
            *(mention.candidate_id for mention in self.mentions),
            *(
                candidate_id
                for group in self.title_groups
                for candidate_id in group.candidate_ids
            ),
            *(item.anchor_candidate_id for item in self.resolver_evidence),
            *(
                candidate_id
                for item in self.resolver_evidence
                for candidate_id in item.candidate_ids
            ),
            *(
                endpoint
                for item in self.bridges
                for endpoint in (
                    item.bridge.source_candidate_id,
                    item.bridge.target_candidate_id,
                )
            ),
            *(
                endpoint
                for edge in self.emitted_edges
                for endpoint in (edge.source, edge.target)
            ),
        }
        _require_candidate_subset(referenced, valid_candidate_ids, "GraphRAG trace")


class ProvenanceBindingTrace(DomainModel):
    output_field: NonEmptyStr
    input_parameter: NonEmptyStr
    binding_kind: NonEmptyStr


class ProvenanceEdgeTrace(DomainModel):
    source: NonEmptyStr
    target: NonEmptyStr
    edge_type: ProvenanceEdgeType
    weight: NonNegativeFiniteFloat
    binding: ProvenanceBindingTrace | None = None
    semantic_rank: PositiveInt | None = None
    semantic_score: FiniteFloat | None = None

    @model_validator(mode="after")
    def _validate_edge(self) -> "ProvenanceEdgeTrace":
        if self.source == self.target:
            raise ValueError("provenance edge cannot be a self loop")
        if self.edge_type is ProvenanceEdgeType.FEEDS and self.binding is None:
            raise ValueError("feeds edge requires a binding")
        if self.edge_type is not ProvenanceEdgeType.FEEDS and self.binding is not None:
            raise ValueError("binding is only valid on feeds edges")
        return self


class StatelessProvenancePathTrace(DomainModel):
    anchor_id: NonEmptyStr
    partner_id: NonEmptyStr
    node_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    path_confidence: FiniteFloat
    binding_valid: StrictBool
    completeness_valid: StrictBool
    lifecycle_valid: StrictBool
    accepted: StrictBool
    rejection_reason: NonEmptyStr | None
    original_partner_rank: PositiveInt
    final_partner_rank: PositiveInt

    @model_validator(mode="after")
    def _validate_path(self) -> "StatelessProvenancePathTrace":
        _require_unique(self.node_ids, "path.node_ids")
        if self.accepted and self.rejection_reason is not None:
            raise ValueError("accepted path cannot have a rejection reason")
        if not self.accepted and self.rejection_reason is None:
            raise ValueError("rejected path requires a reason")
        return self


class StatelessExecutionProvenanceTrace(_NativeTraceModel):
    dense_ranks: tuple[DenseRankTrace, ...]
    seed_candidate_ids: tuple[NonEmptyStr, ...]
    paths: tuple[StatelessProvenancePathTrace, ...]
    edges: tuple[ProvenanceEdgeTrace, ...]
    protected_prefix: tuple[NonEmptyStr, ...]
    exact_dense_fallback: StrictBool
    emitted_edges: tuple[CandidateEdgeTrace, ...]
    scorer_identity: NonEmptyStr
    variant: Literal["typed_beam", "dependency_path", "ppr_steiner"] = "typed_beam"
    trace_kind: Literal["execution_provenance_local"] = "execution_provenance_local"

    @model_validator(mode="after")
    def _validate_trace(self) -> "StatelessExecutionProvenanceTrace":
        _require_unique_dense_ranks(self.dense_ranks)
        _require_unique(self.seed_candidate_ids, "seed_candidate_ids")
        _require_unique(self.protected_prefix, "protected_prefix")
        accepted_count = sum(path.accepted for path in self.paths)
        if self.exact_dense_fallback != (accepted_count == 0):
            raise ValueError("exact fallback state is inconsistent")
        return self

    def validate_candidate_context(self, valid_candidate_ids: frozenset[str]) -> None:
        referenced = {
            *(rank.node_id for rank in self.dense_ranks),
            *self.seed_candidate_ids,
            *self.protected_prefix,
            *(
                endpoint
                for path in self.paths
                for endpoint in (path.anchor_id, path.partner_id)
            ),
            *(
                endpoint
                for edge in self.emitted_edges
                for endpoint in (edge.source, edge.target)
            ),
        }
        _require_candidate_subset(referenced, valid_candidate_ids, "provenance trace")


class ProvenanceRelationAffinityTrace(DomainModel):
    edge_type: ProvenanceEdgeType
    similarity: FiniteFloat
    affinity: NonNegativeFiniteFloat


class ProvenanceTransitionTrace(DomainModel):
    source: NonEmptyStr
    target: NonEmptyStr
    edge_type: ProvenanceEdgeType
    direction: Literal["forward", "reverse"]
    recorded_weight: NonNegativeFiniteFloat
    relation_affinity: NonNegativeFiniteFloat
    probability: NonNegativeFiniteFloat
    cost: NonNegativeFiniteFloat


class ProvenancePprNodeTrace(DomainModel):
    node_id: NonEmptyStr
    teleport: NonNegativeFiniteFloat
    score: NonNegativeFiniteFloat


class ProvenanceCandidatePrizeTrace(DomainModel):
    node_id: NonEmptyStr
    dense_component: NonNegativeFiniteFloat
    ppr_component: NonNegativeFiniteFloat
    prize: NonNegativeFiniteFloat


class ProvenanceSelectedArcTrace(DomainModel):
    source: NonEmptyStr
    target: NonEmptyStr
    edge_type: ProvenanceEdgeType
    direction: Literal["forward", "reverse"]


class ProvenanceSelectionStepTrace(DomainModel):
    anchor_id: NonEmptyStr
    target_id: NonEmptyStr
    path_node_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    transitions: tuple[ProvenanceSelectedArcTrace, ...]
    added_candidate_ids: tuple[NonEmptyStr, ...]
    displaced_candidate_ids: tuple[NonEmptyStr, ...]
    prize_gain: NonNegativeFiniteFloat
    edge_cost: NonNegativeFiniteFloat
    displacement_cost: NonNegativeFiniteFloat
    marginal_gain: FiniteFloat

    @model_validator(mode="after")
    def _validate_step_shape(self) -> "ProvenanceSelectionStepTrace":
        if self.path_node_ids[0] != self.anchor_id:
            raise ValueError("selection path must start at anchor_id")
        if self.path_node_ids[-1] != self.target_id:
            raise ValueError("selection path must end at target_id")
        if len(self.transitions) != len(self.path_node_ids) - 1:
            raise ValueError("selection transition count mismatch")
        _require_unique(self.added_candidate_ids, "added_candidate_ids")
        _require_unique(self.displaced_candidate_ids, "displaced_candidate_ids")
        if set(self.added_candidate_ids) & set(self.displaced_candidate_ids):
            raise ValueError("added and displaced candidates must be disjoint")
        return self


class QueryConditionedExecutionProvenanceTrace(_NativeTraceModel):
    native_graph_node_ids: tuple[NonEmptyStr, ...]
    dense_ranks: tuple[DenseRankTrace, ...]
    relation_description_version: NonEmptyStr
    relations: tuple[ProvenanceRelationAffinityTrace, ...]
    transitions: tuple[ProvenanceTransitionTrace, ...]
    ppr_nodes: tuple[ProvenancePprNodeTrace, ...]
    ppr_iterations: PositiveInt
    ppr_residual: NonNegativeFiniteFloat
    ppr_converged: StrictBool
    candidate_prizes: tuple[ProvenanceCandidatePrizeTrace, ...]
    selected_candidate_ids: tuple[NonEmptyStr, ...]
    connector_node_ids: tuple[NonEmptyStr, ...]
    selection_steps: tuple[ProvenanceSelectionStepTrace, ...]
    selected_native_edges: tuple[ProvenanceEdgeTrace, ...]
    objective: FiniteFloat
    top_k: PositiveInt
    exact_dense_fallback: StrictBool
    emitted_edges: tuple[CandidateEdgeTrace, ...]
    scorer_identity: NonEmptyStr
    variant: Literal["ppr_steiner"] = "ppr_steiner"
    trace_kind: Literal["execution_provenance_subgraph"] = (
        "execution_provenance_subgraph"
    )

    @model_validator(mode="after")
    def _validate_trace(self) -> "QueryConditionedExecutionProvenanceTrace":
        _require_unique(self.native_graph_node_ids, "native_graph_node_ids")
        native_ids = set(self.native_graph_node_ids)
        _require_unique_dense_ranks(self.dense_ranks)

        relation_types = [relation.edge_type for relation in self.relations]
        _require_unique(relation_types, "relations")
        if self.relations and not math.isclose(
            sum(relation.affinity for relation in self.relations),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError("relation affinities must sum to one")
        relation_type_set = set(relation_types)

        transition_keys: set[tuple[str, str, ProvenanceEdgeType, str]] = set()
        transition_row_mass: dict[str, float] = defaultdict(float)
        for transition in self.transitions:
            if (
                transition.source not in native_ids
                or transition.target not in native_ids
                or transition.source == transition.target
            ):
                raise ValueError("invalid transition endpoint")
            if transition.edge_type not in relation_type_set:
                raise ValueError("transition references unknown relation")
            key = (
                transition.source,
                transition.target,
                transition.edge_type,
                transition.direction,
            )
            if key in transition_keys:
                raise ValueError(f"duplicate transition={key}")
            transition_keys.add(key)
            transition_row_mass[transition.source] += transition.probability
        for source, total in transition_row_mass.items():
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
                raise ValueError(
                    f"transition row source={source} does not sum to one"
                )

        ppr_ids = [item.node_id for item in self.ppr_nodes]
        if set(ppr_ids) != native_ids or len(ppr_ids) != len(set(ppr_ids)):
            raise ValueError("PPR nodes must cover native graph exactly")
        if not math.isclose(
            sum(item.teleport for item in self.ppr_nodes), 1.0, abs_tol=1e-8
        ) or not math.isclose(
            sum(item.score for item in self.ppr_nodes), 1.0, abs_tol=1e-8
        ):
            raise ValueError("PPR teleport and score mass must each sum to one")

        _require_unique(self.selected_candidate_ids, "selected_candidate_ids")
        _require_unique(self.connector_node_ids, "connector_node_ids")
        if set(self.connector_node_ids) - native_ids:
            raise ValueError("connector references unknown native node")
        if len(self.selected_candidate_ids) > self.top_k:
            raise ValueError("selected candidates exceed top_k")

        selected_stored_edge_keys: set[tuple[str, str, ProvenanceEdgeType]] = set()
        for step in self.selection_steps:
            if set(step.path_node_ids) - native_ids:
                raise ValueError("selection path references unknown native node")
            for index, arc in enumerate(step.transitions):
                if (
                    arc.source != step.path_node_ids[index]
                    or arc.target != step.path_node_ids[index + 1]
                    or (
                        arc.source,
                        arc.target,
                        arc.edge_type,
                        arc.direction,
                    )
                    not in transition_keys
                ):
                    raise ValueError("selected transition is not a traced path arc")
                selected_stored_edge_keys.add(
                    (arc.source, arc.target, arc.edge_type)
                    if arc.direction == "forward"
                    else (arc.target, arc.source, arc.edge_type)
                )
            if not set(step.added_candidate_ids).issubset(
                self.selected_candidate_ids
            ):
                raise ValueError("added candidates must be selected")

        selected_context = set(self.selected_candidate_ids) | set(
            self.connector_node_ids
        )
        if set(self.connector_node_ids) & set(self.selected_candidate_ids):
            raise ValueError("connectors cannot be candidate nodes")
        selected_edge_keys: set[tuple[str, str, ProvenanceEdgeType]] = set()
        adjacency: dict[str, set[str]] = defaultdict(set)
        for edge in self.selected_native_edges:
            if edge.source not in selected_context or edge.target not in selected_context:
                raise ValueError("selected edge leaves selected context")
            key = (edge.source, edge.target, edge.edge_type)
            if key in selected_edge_keys:
                raise ValueError(f"duplicate selected edge={key}")
            selected_edge_keys.add(key)
            adjacency[edge.source].add(edge.target)
            adjacency[edge.target].add(edge.source)
        if selected_context:
            seen = {next(iter(selected_context))}
            queue = deque(seen)
            while queue:
                current = queue.popleft()
                for neighbor in adjacency.get(current, set()):
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)
            if seen != selected_context:
                raise ValueError("selected subgraph is disconnected")
        if selected_edge_keys != selected_stored_edge_keys:
            raise ValueError("selected native edge orientation is inconsistent")

        if self.exact_dense_fallback != (not self.selection_steps):
            raise ValueError("fallback/intervention state is inconsistent")
        if self.exact_dense_fallback and (
            self.selected_candidate_ids
            or self.connector_node_ids
            or self.selected_native_edges
        ):
            raise ValueError("fallback retains a selected subgraph")
        return self

    def validate_candidate_context(self, valid_candidate_ids: frozenset[str]) -> None:
        native_ids = set(self.native_graph_node_ids)
        if not valid_candidate_ids.issubset(native_ids):
            raise ValueError("candidate is missing from native graph")
        dense_ids = [rank.node_id for rank in self.dense_ranks]
        if set(dense_ids) != valid_candidate_ids or len(dense_ids) != len(
            set(dense_ids)
        ):
            raise ValueError("dense ranks must cover candidates exactly")
        prize_ids = [item.node_id for item in self.candidate_prizes]
        if set(prize_ids) != valid_candidate_ids or len(prize_ids) != len(
            set(prize_ids)
        ):
            raise ValueError("candidate prizes must cover candidates exactly")
        _require_candidate_subset(
            set(self.selected_candidate_ids), valid_candidate_ids, "selected candidates"
        )
        if set(self.connector_node_ids) & set(valid_candidate_ids):
            raise ValueError("connector nodes cannot be candidate nodes")
        referenced = {
            *(
                candidate_id
                for step in self.selection_steps
                for candidate_id in (
                    *step.added_candidate_ids,
                    *step.displaced_candidate_ids,
                )
            ),
            *(
                endpoint
                for edge in self.emitted_edges
                for endpoint in (edge.source, edge.target)
            ),
        }
        _require_candidate_subset(referenced, valid_candidate_ids, "subgraph trace")


class ProvenancePathTrace(DomainModel):
    node_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    score: FiniteFloat
    semantic_relevance: FiniteFloat
    binding_consistency: FiniteFloat
    provenance_completeness: FiniteFloat
    explicit_grounding: FiniteFloat
    path_length_penalty: FiniteFloat
    invalidation_penalty: FiniteFloat


class ProvenanceStructuredTransitionTrace(DomainModel):
    source_id: NonEmptyStr
    target_id: NonEmptyStr
    probability: NonNegativeFiniteFloat
    original_source_rank: NonNegativeInt
    original_target_rank: NonNegativeInt
    final_target_rank: NonNegativeInt
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
    node_ids: tuple[NonEmptyStr, ...]
    paths: tuple[ProvenancePathTrace, ...]
    edges: tuple[ProvenanceEdgeTrace, ...]
    structured_transitions: tuple[ProvenanceStructuredTransitionTrace, ...] = ()
    abstained_source_ids: tuple[NonEmptyStr, ...] = ()
    structured_promotion_enabled: StrictBool = True
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
            edge_key = (edge.source, edge.target, edge.edge_type.value)
            if edge_key in seen_edges:
                raise ValueError(f"duplicate edge={edge_key}")
            seen_edges.add(edge_key)
            adjacent_pairs.add(frozenset((edge.source, edge.target)))

        seen_paths: set[tuple[str, ...]] = set()
        for path in self.paths:
            if path.node_ids in seen_paths:
                raise ValueError(f"duplicate path={list(path.node_ids)}")
            seen_paths.add(path.node_ids)
            if len(path.node_ids) != len(set(path.node_ids)):
                raise ValueError("path.node_ids contains duplicates")
            unknown_nodes = sorted(set(path.node_ids) - node_id_set)
            if unknown_nodes:
                raise ValueError(f"path references unknown nodes={unknown_nodes}")
            for source, target in zip(path.node_ids, path.node_ids[1:]):
                if frozenset((source, target)) not in adjacent_pairs:
                    raise ValueError(
                        f"path step {source}->{target} has no traced edge"
                    )

        _require_unique(self.abstained_source_ids, "abstained_source_ids")
        seen_transitions: set[tuple[str, str]] = set()
        for transition in self.structured_transitions:
            transition_key = (transition.source_id, transition.target_id)
            if transition_key in seen_transitions:
                raise ValueError(
                    "duplicate structured_transition="
                    f"{transition.source_id}->{transition.target_id}"
                )
            seen_transitions.add(transition_key)
        return self


NativeRetrievalTrace: TypeAlias = Annotated[
    EntitySearchTrace
    | GraphRAGTrace
    | ExecutionProvenanceTrace
    | StatelessExecutionProvenanceTrace
    | QueryConditionedExecutionProvenanceTrace,
    Field(discriminator="trace_kind"),
]


class RetrievalTrace(DomainModel):
    retrieved_edges: tuple[GraphEdge, ...] = ()
    native_trace: NativeRetrievalTrace | None = None


class RetrievalMethodResult(DomainModel):
    ranked_nodes: tuple[RankedNode, ...]
    trace: RetrievalTrace = RetrievalTrace()


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


def _require_unique(values: Sequence[object], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} contains duplicates")


def _require_unique_dense_ranks(ranks: tuple[DenseRankTrace, ...]) -> None:
    _require_unique(tuple(rank.node_id for rank in ranks), "dense_ranks.node_id")
    _require_unique(tuple(rank.dense_rank for rank in ranks), "dense_ranks.dense_rank")
    _require_unique(tuple(rank.final_rank for rank in ranks), "dense_ranks.final_rank")


def _require_candidate_subset(
    referenced: set[str], valid: frozenset[str], component: str
) -> None:
    unknown = sorted(referenced - valid)
    if unknown:
        raise ValueError(f"{component} references unknown candidates={unknown}")
