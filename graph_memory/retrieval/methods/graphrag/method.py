from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from graph_memory.graphs.contracts import GraphEdge
from graph_memory.retrieval.contracts import (
    CandidateEdgeTrace,
    DenseRankTrace,
    GraphRAGBridgeTrace,
    GraphRAGTrace,
    RankedNode,
    RetrievalMethodResult,
    RetrievalTrace,
)
from graph_memory.retrieval.methods.flat.dense import DenseTaskRetriever
from graph_memory.retrieval.methods.graphrag.config import GraphRAGConfig
from graph_memory.retrieval.methods.graphrag.index import link_query_entities
from graph_memory.retrieval.methods.graphrag.sentence_resolver import (
    GraphRAGSentenceResolver,
    SentenceResolutionInput,
)
from graph_memory.retrieval.requests import (
    GraphRAGCandidateBridge,
    GraphRAGEntityMention,
    GraphRAGRequest,
    GraphRAGResolverEvidence,
    RankingMethodRequest,
    TextRankingRequest,
)


@dataclass(frozen=True)
class GraphRAGMethod:
    dense_ranker: DenseTaskRetriever
    config: GraphRAGConfig = GraphRAGConfig()
    sentence_resolver: GraphRAGSentenceResolver | None = None
    name: str = "graphrag"

    def __post_init__(self) -> None:
        if self.sentence_resolver is None:
            object.__setattr__(
                self,
                "sentence_resolver",
                GraphRAGSentenceResolver(
                    encoder=self.dense_ranker.encoder,
                    query_prefix=self.dense_ranker.config.query_prefix,
                    passage_prefix=self.dense_ranker.config.passage_prefix,
                    batch_size=self.dense_ranker.config.batch_size,
                    min_score_margin=self.config.min_sentence_score_margin,
                ),
            )
        assert self.sentence_resolver is not None
        if self.sentence_resolver.encoder is not self.dense_ranker.encoder:
            raise ValueError(
                "GraphRAG Dense ranker and sentence resolver must share one encoder."
            )

    def rank_task(
        self,
        request: RankingMethodRequest,
        *,
        top_k: int,
    ) -> RetrievalMethodResult:
        if not isinstance(request, GraphRAGRequest):
            raise TypeError(
                f"{self.name} requires GraphRAGRequest, got {type(request).__name__}."
            )
        dense_ranked = self.dense_ranker.rank(
            TextRankingRequest(
                task_id=request.task_id,
                query_text=request.query_text,
                candidates=request.candidates,
            )
        )
        dense_rank = {
            node.node_id: index for index, node in enumerate(dense_ranked, start=1)
        }
        seed_candidate_ids = tuple(
            node.node_id for node in dense_ranked[: self.config.seed_top_s]
        )
        linked_entity_ids = link_query_entities(
            request.query_text, request.knowledge_graph
        )
        resolution_inputs = _resolution_inputs(
            request,
            anchor_ids=seed_candidate_ids,
            max_document_frequency_ratio=(
                self.config.max_entity_document_frequency_ratio
            ),
            max_groups_per_anchor=self.config.max_entity_groups_per_anchor,
        )
        assert self.sentence_resolver is not None
        resolver_evidence = self.sentence_resolver.resolve_many(
            request,
            resolution_inputs,
            original_dense_rank=dense_rank,
        )
        bridges = _candidate_bridges(
            request,
            resolver_evidence=resolver_evidence,
            linked_entity_ids=frozenset(linked_entity_ids),
            dense_rank=dense_rank,
        )
        outcomes = _select_bridge_proposals(
            bridges,
            dense_rank=dense_rank,
            config=self.config,
        )
        final_ids, moved = _stable_insert(
            [node.node_id for node in dense_ranked],
            outcomes=outcomes,
            dense_rank=dense_rank,
            preserve_dense_top_n=self.config.preserve_dense_top_n,
        )
        actual_bridges = tuple(
            bridge for bridge, _old_rank, _new_rank in moved
        )
        for bridge in tuple(outcomes):
            accepted, reason = outcomes[bridge]
            if accepted and bridge not in actual_bridges:
                outcomes[bridge] = (False, reason or "no_effective_insertion")
        if not moved:
            ranked_nodes = dense_ranked
        else:
            score_slots = [node.score for node in dense_ranked]
            ranked_nodes = [
                RankedNode(node_id=node_id, score=score_slots[index])
                for index, node_id in enumerate(final_ids)
            ]
        final_rank = {
            node.node_id: index for index, node in enumerate(ranked_nodes, start=1)
        }
        emitted_bridges = tuple(
            bridge
            for bridge, _old_rank, _new_rank in moved
            if bridge.source_candidate_id in set(final_ids[:top_k])
            and bridge.target_candidate_id in set(final_ids[:top_k])
        )
        retrieved_edges = [_bridge_edge(bridge) for bridge in emitted_bridges]
        trace = GraphRAGTrace(
            dense_ranks=tuple(
                DenseRankTrace(
                    node_id=node.node_id,
                    dense_rank=dense_rank[node.node_id],
                    dense_score=node.score,
                    final_rank=final_rank[node.node_id],
                )
                for node in dense_ranked
            ),
            seed_candidate_ids=seed_candidate_ids,
            linked_entity_ids=linked_entity_ids,
            mentions=request.knowledge_graph.mentions,
            title_groups=request.knowledge_graph.title_groups,
            resolver_evidence=resolver_evidence,
            bridges=tuple(
                GraphRAGBridgeTrace(
                    bridge=bridge,
                    accepted=outcomes[bridge][0] and bridge in actual_bridges,
                    rejection_reason=(
                        None
                        if outcomes[bridge][0] and bridge in actual_bridges
                        else outcomes[bridge][1]
                    ),
                    original_partner_rank=dense_rank[bridge.target_candidate_id],
                    final_partner_rank=final_rank[bridge.target_candidate_id],
                )
                for bridge in bridges
            ),
            protected_prefix=tuple(
                node.node_id
                for node in dense_ranked[: self.config.preserve_dense_top_n]
            ),
            exact_dense_fallback=not moved,
            emitted_edges=tuple(
                CandidateEdgeTrace(
                    source=bridge.source_candidate_id,
                    target=bridge.target_candidate_id,
                    edge_type="bridge_to",
                    confidence=bridge.confidence,
                )
                for bridge in emitted_bridges
            ),
        )
        return RetrievalMethodResult(
            ranked_nodes=tuple(ranked_nodes),
            trace=RetrievalTrace(
                retrieved_edges=tuple(retrieved_edges),
                native_trace=trace,
            ),
        )


def _resolution_inputs(
    request: GraphRAGRequest,
    *,
    anchor_ids: tuple[str, ...],
    max_document_frequency_ratio: float,
    max_groups_per_anchor: int,
) -> tuple[SentenceResolutionInput, ...]:
    groups_by_entity = {
        group.entity_id: group for group in request.knowledge_graph.title_groups
    }
    body_entities_by_candidate: dict[str, set[str]] = defaultdict(set)
    for mention in request.knowledge_graph.mentions:
        if mention.mention_type == "MENTIONS" and mention.alias_confidence > 0.0:
            body_entities_by_candidate[mention.candidate_id].add(mention.entity_id)
    mention_confidence = {
        (mention.candidate_id, mention.entity_id): mention.mention_confidence
        for mention in request.knowledge_graph.mentions
        if mention.mention_type == "MENTIONS"
    }
    result: list[SentenceResolutionInput] = []
    for anchor_id in anchor_ids:
        eligible = []
        for entity_id in body_entities_by_candidate.get(anchor_id, set()):
            group = groups_by_entity.get(entity_id)
            if (
                group is None
                or group.document_frequency_ratio > max_document_frequency_ratio
            ):
                continue
            eligible.append(group)
        eligible.sort(
            key=lambda group: (
                -mention_confidence.get((anchor_id, group.entity_id), 0.0),
                group.document_frequency_ratio,
                group.entity_id,
            )
        )
        result.extend(
            SentenceResolutionInput(anchor_id, group)
            for group in eligible[:max_groups_per_anchor]
        )
    return tuple(result)


def _candidate_bridges(
    request: GraphRAGRequest,
    *,
    resolver_evidence: tuple[GraphRAGResolverEvidence, ...],
    linked_entity_ids: frozenset[str],
    dense_rank: dict[str, int],
) -> tuple[GraphRAGCandidateBridge, ...]:
    mentions_by_key: dict[tuple[str, str, str], GraphRAGEntityMention] = {
        (mention.candidate_id, mention.entity_id, mention.mention_type): mention
        for mention in request.knowledge_graph.mentions
    }
    result: list[GraphRAGCandidateBridge] = []
    for evidence in resolver_evidence:
        target_id = evidence.selected_candidate_id
        if not evidence.accepted or target_id is None or evidence.top1_score is None:
            continue
        source_mention = mentions_by_key.get(
            (evidence.anchor_candidate_id, evidence.entity_id, "MENTIONS")
        )
        target_mention = mentions_by_key.get(
            (target_id, evidence.entity_id, "TITLE_ENTITY")
        ) or mentions_by_key.get(
            (target_id, evidence.entity_id, "MENTIONS")
        )
        if source_mention is None or target_mention is None:
            continue
        seed_confidence = (
            1.0
            if evidence.entity_id in linked_entity_ids
            else 1.0 / math.log2(2 + dense_rank[evidence.anchor_candidate_id])
        )
        confidence = (
            seed_confidence
            * source_mention.mention_confidence
            * target_mention.mention_confidence
        ) ** (1.0 / 3.0)
        result.append(
            GraphRAGCandidateBridge(
                source_candidate_id=evidence.anchor_candidate_id,
                target_candidate_id=target_id,
                bridge_entity_id=evidence.entity_id,
                confidence=confidence,
                resolver_score=evidence.top1_score,
                resolver_margin=evidence.score_margin,
                construction_reason=(
                    "body_mention_to_resolved_title_entity"
                    if target_mention.mention_type == "TITLE_ENTITY"
                    else "shared_text_entity_to_resolved_candidate"
                ),
            )
        )
    return tuple(
        sorted(
            result,
            key=lambda bridge: (
                dense_rank[bridge.source_candidate_id],
                -bridge.confidence,
                dense_rank[bridge.target_candidate_id],
                bridge.target_candidate_id,
                bridge.bridge_entity_id,
            ),
        )
    )


def _select_bridge_proposals(
    bridges: tuple[GraphRAGCandidateBridge, ...],
    *,
    dense_rank: dict[str, int],
    config: GraphRAGConfig,
) -> dict[GraphRAGCandidateBridge, tuple[bool, str | None]]:
    outcomes: dict[GraphRAGCandidateBridge, tuple[bool, str | None]] = {}
    eligible_by_anchor: dict[str, list[GraphRAGCandidateBridge]] = defaultdict(list)
    for bridge in bridges:
        partner_rank = dense_rank[bridge.target_candidate_id]
        anchor_rank = dense_rank[bridge.source_candidate_id]
        if bridge.confidence < config.min_bridge_confidence:
            outcomes[bridge] = (False, "below_bridge_confidence")
        elif partner_rank <= config.preserve_dense_top_n:
            outcomes[bridge] = (False, "protected_partner")
        elif partner_rank <= anchor_rank:
            outcomes[bridge] = (False, "partner_not_after_anchor")
        else:
            eligible_by_anchor[bridge.source_candidate_id].append(bridge)
    anchor_winners: list[GraphRAGCandidateBridge] = []
    for anchor_id, candidates in eligible_by_anchor.items():
        ordered = sorted(
            candidates,
            key=lambda bridge: (
                -bridge.confidence,
                dense_rank[bridge.target_candidate_id],
                bridge.target_candidate_id,
                bridge.bridge_entity_id,
            ),
        )
        anchor_winners.append(ordered[0])
        for bridge in ordered[1:]:
            outcomes[bridge] = (False, "lower_confidence_for_anchor")
    by_partner: dict[str, list[GraphRAGCandidateBridge]] = defaultdict(list)
    for bridge in anchor_winners:
        by_partner[bridge.target_candidate_id].append(bridge)
    for candidates in by_partner.values():
        ordered = sorted(
            candidates,
            key=lambda bridge: (
                -bridge.confidence,
                dense_rank[bridge.source_candidate_id],
                dense_rank[bridge.target_candidate_id],
                bridge.target_candidate_id,
            ),
        )
        outcomes[ordered[0]] = (True, None)
        for bridge in ordered[1:]:
            outcomes[bridge] = (False, "partner_conflict")
    return outcomes


def _stable_insert(
    dense_ids: list[str],
    *,
    outcomes: dict[GraphRAGCandidateBridge, tuple[bool, str | None]],
    dense_rank: dict[str, int],
    preserve_dense_top_n: int,
) -> tuple[list[str], list[tuple[GraphRAGCandidateBridge, int, int]]]:
    final_ids = list(dense_ids)
    moved: list[tuple[GraphRAGCandidateBridge, int, int]] = []
    accepted = sorted(
        (bridge for bridge, outcome in outcomes.items() if outcome[0]),
        key=lambda bridge: (
            dense_rank[bridge.source_candidate_id],
            dense_rank[bridge.target_candidate_id],
            bridge.target_candidate_id,
        ),
    )
    for bridge in accepted:
        anchor_position = final_ids.index(bridge.source_candidate_id)
        partner_position = final_ids.index(bridge.target_candidate_id)
        insertion_position = max(preserve_dense_top_n, anchor_position + 1)
        if partner_position <= insertion_position:
            outcomes[bridge] = (False, "no_effective_insertion")
            continue
        final_ids.pop(partner_position)
        final_ids.insert(insertion_position, bridge.target_candidate_id)
        moved.append((bridge, partner_position + 1, insertion_position + 1))
    return final_ids, moved


def _bridge_edge(bridge: GraphRAGCandidateBridge) -> GraphEdge:
    return GraphEdge(
        source=bridge.source_candidate_id,
        target=bridge.target_candidate_id,
        edge_type="bridge_to",
        weight=bridge.confidence,
        directed=True,
    )


__all__ = ["GraphRAGMethod"]
