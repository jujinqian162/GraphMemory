from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal, Protocol, TypeAlias

from pydantic import Field, StrictBool, model_validator

from graph_memory.contracts.model import (
    DomainModel,
    FiniteFloat,
    NonEmptyStr,
    NonNegativeFiniteFloat,
    PositiveFiniteFloat,
    PositiveInt,
)
from graph_memory.graphs.contracts import GraphEdge
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
    rejection_reason: NonEmptyStr | None = None
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


class ProvenanceGraphEdgeTrace(DomainModel):
    edge_id: NonEmptyStr
    source_node_id: NonEmptyStr
    target_node_id: NonEmptyStr
    relation: NonEmptyStr

    @model_validator(mode="after")
    def _validate_edge(self) -> "ProvenanceGraphEdgeTrace":
        if self.source_node_id == self.target_node_id:
            raise ValueError("provenance graph edge cannot be a self edge")
        return self


class ProvenancePathProposalTrace(DomainModel):
    anchor_candidate_id: NonEmptyStr
    partner_candidate_id: NonEmptyStr
    path_node_ids: tuple[NonEmptyStr, ...] = Field(min_length=2)
    path_relations: tuple[NonEmptyStr, ...] = Field(min_length=1)
    path_edge_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    traversed_reverse: tuple[StrictBool, ...] = Field(min_length=1)
    accepted: StrictBool
    rejection_reason: NonEmptyStr | None = None
    original_anchor_rank: PositiveInt
    original_partner_rank: PositiveInt
    final_partner_rank: PositiveInt

    @model_validator(mode="after")
    def _validate_proposal(self) -> "ProvenancePathProposalTrace":
        if self.anchor_candidate_id == self.partner_candidate_id:
            raise ValueError("provenance proposal endpoints must differ")
        if self.path_node_ids[0] != self.anchor_candidate_id:
            raise ValueError("provenance proposal path must start at its anchor")
        if self.path_node_ids[-1] != self.partner_candidate_id:
            raise ValueError("provenance proposal path must end at its partner")
        expected_hops = len(self.path_node_ids) - 1
        if len(self.path_relations) != expected_hops:
            raise ValueError("provenance proposal relation count must match path")
        if len(self.path_edge_ids) != expected_hops:
            raise ValueError("provenance proposal edge count must match path")
        if len(self.traversed_reverse) != expected_hops:
            raise ValueError("provenance proposal direction count must match path")
        if self.accepted == (self.rejection_reason is not None):
            raise ValueError("provenance proposal outcome is inconsistent")
        return self


class ProvenancePathTrace(_NativeTraceModel):
    dense_ranks: tuple[DenseRankTrace, ...]
    seed_candidate_ids: tuple[NonEmptyStr, ...]
    graph_edges: tuple[ProvenanceGraphEdgeTrace, ...]
    proposals: tuple[ProvenancePathProposalTrace, ...]
    protected_prefix: tuple[NonEmptyStr, ...]
    emitted_edges: tuple[CandidateEdgeTrace, ...]
    seed_top_s: PositiveInt
    max_path_hops: PositiveInt
    max_partners_per_anchor: PositiveInt
    max_expansions: PositiveInt
    exact_dense_fallback: StrictBool
    trace_kind: Literal["provenance_path"] = "provenance_path"

    @model_validator(mode="after")
    def _validate_trace(self) -> "ProvenancePathTrace":
        _require_unique_dense_ranks(self.dense_ranks)
        _require_unique(self.seed_candidate_ids, "seed_candidate_ids")
        _require_unique(self.protected_prefix, "protected_prefix")
        accepted = any(proposal.accepted for proposal in self.proposals)
        if self.exact_dense_fallback == accepted:
            raise ValueError("provenance path fallback state is inconsistent")
        edge_keys = [
            (edge.source, edge.target, edge.edge_type)
            for edge in self.emitted_edges
        ]
        if len(edge_keys) != len(set(edge_keys)):
            raise ValueError("provenance path emitted edges must be unique")
        return self

    def validate_candidate_context(self, valid_candidate_ids: frozenset[str]) -> None:
        referenced = {
            *(rank.node_id for rank in self.dense_ranks),
            *self.seed_candidate_ids,
            *self.protected_prefix,
            *(
                candidate_id
                for proposal in self.proposals
                for candidate_id in (
                    proposal.anchor_candidate_id,
                    proposal.partner_candidate_id,
                )
            ),
            *(
                endpoint
                for edge in self.emitted_edges
                for endpoint in (edge.source, edge.target)
            ),
        }
        _require_candidate_subset(referenced, valid_candidate_ids, "provenance path trace")


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



NativeRetrievalTrace: TypeAlias = Annotated[
    EntitySearchTrace | GraphRAGTrace | ProvenancePathTrace,
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
