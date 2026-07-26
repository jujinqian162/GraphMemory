"""The single non-trained EPGM provenance retriever.

One class, one registry id (``execution_provenance_retriever``), two frozen
presets selected by ``EpgmRetrieverConfig.variant``:

* ``typed_beam`` (default) - the reported "EPGM (non-trained)" method.
* ``dependency_path`` - a restriction of the same search kept only as an
  ablation of the default.

``rank_task`` is the registry/evaluation entry point. ``rank`` exposes the
richer per-node result used by the standalone real-trace runner.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import cast

from graph_memory.contracts.graphs import GraphEdge
from graph_memory.graphs.provenance import (
    ExecutionProvenanceEdge,
    ProvenanceEdgeType,
)
from graph_memory.retrieval.contracts import (
    CandidateEdgeTrace,
    DenseRankTrace,
    ProvenanceBindingTrace,
    ProvenanceEdgeTrace,
    RankedNode,
    RetrievalMethodResult,
    RetrievalTrace,
    StatelessExecutionProvenanceTrace,
    StatelessProvenancePathTrace,
)
from graph_memory.retrieval.methods.epgm.config import EpgmRetrieverConfig
from graph_memory.retrieval.methods.epgm.search import EpgmPath, search_epgm_paths
from graph_memory.retrieval.methods.flat.dense import DenseTaskRetriever
from graph_memory.retrieval.requests import (
    ExecutionProvenanceRankingRequest,
    RankingMethodRequest,
    TextRankingRequest,
)


@dataclass(frozen=True)
class EpgmRankedNode:
    node_id: str
    score: float
    dense_rank: int
    dense_score: float
    graph_score: float
    best_path: EpgmPath | None


@dataclass(frozen=True)
class EpgmRetrievalResult:
    ranked_nodes: tuple[EpgmRankedNode, ...]
    seed_ids: tuple[str, ...]
    paths: tuple[EpgmPath, ...]


@dataclass(frozen=True)
class EpgmRetriever:
    """Dense seed + bounded typed path reranking. No learned parameters."""

    dense_ranker: DenseTaskRetriever
    config: EpgmRetrieverConfig = EpgmRetrieverConfig()
    name: str = "execution_provenance_retriever"
    display_name: str = "EPGM (non-trained)"

    @property
    def variant(self) -> str:
        return self.config.variant

    def rank_task(
        self,
        request: RankingMethodRequest,
        *,
        top_k: int,
    ) -> RetrievalMethodResult:
        if not isinstance(request, ExecutionProvenanceRankingRequest):
            raise TypeError(
                f"{self.name} requires ExecutionProvenanceRankingRequest, "
                f"got {type(request).__name__}."
            )
        dense_ranked = self.dense_ranker.rank(
            TextRankingRequest(request.task_id, request.query_text, request.candidates)
        )
        dense_rank = {
            node.node_id: index for index, node in enumerate(dense_ranked, start=1)
        }
        seed_relevance = _normalized_relevance(dense_ranked)
        seed_ids = tuple(
            node.node_id for node in dense_ranked[: self.config.seed_top_s]
        )
        paths = search_epgm_paths(
            request.graph,
            seed_ids,
            seed_relevance,
            candidate_ids=frozenset(dense_rank),
            config=self.config,
        )
        best_by_target = _best_path_by_target(paths)
        outcomes = _select_paths(paths, dense_rank=dense_rank, config=self.config)

        if self.config.fusion == "stable_insert":
            ranked_nodes, promoted = _stable_insert_ranking(
                dense_ranked,
                outcomes=outcomes,
                dense_rank=dense_rank,
                preserve_dense_top_n=self.config.preserve_dense_top_n,
            )
            exact_dense_fallback = not promoted
        else:
            ranked_nodes = _additive_ranking(
                dense_ranked,
                seed_relevance=seed_relevance,
                best_by_target=best_by_target,
                graph_weight=self.config.graph_weight,
            )
            promoted = tuple(
                path for path, (accepted, _reason) in outcomes.items() if accepted
            )
            exact_dense_fallback = [node.node_id for node in ranked_nodes] == [
                node.node_id for node in dense_ranked
            ]

        final_rank = {
            node.node_id: index for index, node in enumerate(ranked_nodes, start=1)
        }
        top_ids = {node.node_id for node in ranked_nodes[:top_k]}
        emitted = tuple(
            path
            for path in promoted
            if path.seed_id in top_ids and path.target_id in top_ids
        )
        return RetrievalMethodResult(
            ranked_nodes=ranked_nodes,
            trace=RetrievalTrace(
                retrieved_edges=_logical_edges(emitted),
                native_trace=StatelessExecutionProvenanceTrace(
                    dense_ranks=tuple(
                        DenseRankTrace(
                            node_id=node.node_id,
                            dense_rank=dense_rank[node.node_id],
                            dense_score=node.score,
                            final_rank=final_rank[node.node_id],
                        )
                        for node in dense_ranked
                    ),
                    seed_candidate_ids=seed_ids,
                    paths=tuple(
                        StatelessProvenancePathTrace(
                            anchor_id=path.seed_id,
                            partner_id=path.target_id,
                            node_ids=path.node_ids,
                            path_confidence=path.score,
                            binding_valid=path.gate.binding_valid,
                            completeness_valid=path.gate.completeness_valid,
                            lifecycle_valid=path.gate.lifecycle_valid,
                            accepted=outcomes[path][0],
                            rejection_reason=outcomes[path][1],
                            original_partner_rank=dense_rank[path.target_id],
                            final_partner_rank=final_rank[path.target_id],
                        )
                        for path in paths
                    ),
                    edges=_edge_traces(paths, request),
                    protected_prefix=tuple(
                        node.node_id
                        for node in dense_ranked[: self.config.preserve_dense_top_n]
                    ),
                    exact_dense_fallback=exact_dense_fallback,
                    emitted_edges=tuple(
                        CandidateEdgeTrace(
                            source=_logical_endpoints(path)[0],
                            target=_logical_endpoints(path)[1],
                            edge_type=_logical_edge_type(path),
                            confidence=path.score,
                        )
                        for path in emitted
                    ),
                    scorer_identity=_scorer_identity(request),
                    variant=self.config.variant,
                ),
            ),
        )

    def rank(self, request: ExecutionProvenanceRankingRequest) -> EpgmRetrievalResult:
        """Per-node view with dense/graph score decomposition and best paths."""

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
        dense_score = {node.node_id: node.score for node in dense_ranked}
        seed_relevance = _normalized_relevance(dense_ranked)
        seed_ids = tuple(
            node.node_id for node in dense_ranked[: self.config.seed_top_s]
        )
        paths = search_epgm_paths(
            request.graph,
            seed_ids,
            seed_relevance,
            candidate_ids=frozenset(dense_score),
            config=self.config,
        )
        best_by_target = _best_path_by_target(paths)
        fused = _additive_ranking(
            dense_ranked,
            seed_relevance=seed_relevance,
            best_by_target=best_by_target,
            graph_weight=self.config.graph_weight,
        )
        ranked = tuple(
            EpgmRankedNode(
                node_id=node.node_id,
                score=node.score,
                dense_rank=dense_rank[node.node_id],
                dense_score=dense_score[node.node_id],
                graph_score=(
                    best_by_target[node.node_id].score
                    if node.node_id in best_by_target
                    else 0.0
                ),
                best_path=best_by_target.get(node.node_id),
            )
            for node in fused
        )
        return EpgmRetrievalResult(
            ranked_nodes=ranked, seed_ids=seed_ids, paths=paths
        )


def _normalized_relevance(dense_ranked: list[RankedNode]) -> dict[str, float]:
    if not dense_ranked:
        return {}
    scores = [node.score for node in dense_ranked]
    lo = min(scores)
    span = max(scores) - lo
    if span <= 0.0:
        return {node.node_id: 1.0 for node in dense_ranked}
    return {node.node_id: (node.score - lo) / span for node in dense_ranked}


def _best_path_by_target(paths: tuple[EpgmPath, ...]) -> dict[str, EpgmPath]:
    best: dict[str, EpgmPath] = {}
    for path in paths:
        current = best.get(path.target_id)
        if current is None or path.score > current.score:
            best[path.target_id] = path
    return best


def _additive_ranking(
    dense_ranked: list[RankedNode],
    *,
    seed_relevance: dict[str, float],
    best_by_target: dict[str, EpgmPath],
    graph_weight: float,
) -> list[RankedNode]:
    """Additive fusion: graph propagation may lift a node, never demote it."""

    dense_rank = {
        node.node_id: index for index, node in enumerate(dense_ranked, start=1)
    }
    fused = [
        RankedNode(
            node.node_id,
            seed_relevance.get(node.node_id, 0.0)
            + graph_weight
            * (
                best_by_target[node.node_id].score
                if node.node_id in best_by_target
                else 0.0
            ),
        )
        for node in dense_ranked
    ]
    fused.sort(key=lambda node: (-node.score, dense_rank[node.node_id], node.node_id))
    return fused


def _select_paths(
    paths: tuple[EpgmPath, ...],
    *,
    dense_rank: dict[str, int],
    config: EpgmRetrieverConfig,
) -> dict[EpgmPath, tuple[bool, str | None]]:
    """One accepted partner per anchor and per partner, with gate reasons."""

    outcomes: dict[EpgmPath, tuple[bool, str | None]] = {}
    eligible_by_anchor: dict[str, list[EpgmPath]] = defaultdict(list)
    for path in paths:
        anchor_rank = dense_rank[path.seed_id]
        partner_rank = dense_rank[path.target_id]
        if not path.gate.valid:
            outcomes[path] = (False, path.gate.rejection_reason or "invalid_path")
        elif path.score <= 0.0:
            # A zero-relevance seed carries no evidence; emitting its edge
            # would only add noise to edge precision.
            outcomes[path] = (False, "no_path_evidence")
        elif path.score < config.min_path_confidence:
            outcomes[path] = (False, "below_path_confidence")
        elif config.fusion == "stable_insert" and (
            partner_rank <= config.preserve_dense_top_n
        ):
            outcomes[path] = (False, "protected_partner")
        elif config.fusion == "stable_insert" and partner_rank <= anchor_rank:
            outcomes[path] = (False, "partner_not_after_anchor")
        else:
            eligible_by_anchor[path.seed_id].append(path)

    winners: list[EpgmPath] = []
    for candidates in eligible_by_anchor.values():
        ordered = sorted(
            candidates,
            key=lambda path: (
                -path.score,
                dense_rank[path.target_id],
                path.target_id,
                path.node_ids,
            ),
        )
        winners.append(ordered[0])
        for path in ordered[1:]:
            outcomes[path] = (False, "lower_confidence_for_anchor")

    by_partner: dict[str, list[EpgmPath]] = defaultdict(list)
    for path in winners:
        by_partner[path.target_id].append(path)
    for candidates in by_partner.values():
        ordered = sorted(
            candidates,
            key=lambda path: (
                -path.score,
                dense_rank[path.seed_id],
                dense_rank[path.target_id],
                path.target_id,
            ),
        )
        outcomes[ordered[0]] = (True, None)
        for path in ordered[1:]:
            outcomes[path] = (False, "partner_conflict")
    return outcomes


def _stable_insert_ranking(
    dense_ranked: list[RankedNode],
    *,
    outcomes: dict[EpgmPath, tuple[bool, str | None]],
    dense_rank: dict[str, int],
    preserve_dense_top_n: int,
) -> tuple[list[RankedNode], tuple[EpgmPath, ...]]:
    """Reorder node ids only; the dense score multiset is preserved exactly."""

    final_ids = [node.node_id for node in dense_ranked]
    promoted: list[EpgmPath] = []
    accepted = sorted(
        (path for path, outcome in outcomes.items() if outcome[0]),
        key=lambda path: (
            dense_rank[path.seed_id],
            dense_rank[path.target_id],
            path.target_id,
        ),
    )
    for path in accepted:
        anchor_position = final_ids.index(path.seed_id)
        partner_position = final_ids.index(path.target_id)
        insertion_position = max(preserve_dense_top_n, anchor_position + 1)
        if partner_position <= insertion_position:
            outcomes[path] = (False, "no_effective_insertion")
            continue
        final_ids.pop(partner_position)
        final_ids.insert(insertion_position, path.target_id)
        promoted.append(path)
    if not promoted:
        return dense_ranked, ()
    score_slots = [node.score for node in dense_ranked]
    ranked = [
        RankedNode(node_id, score_slots[index])
        for index, node_id in enumerate(final_ids)
    ]
    return ranked, tuple(promoted)


def _logical_edge_type(path: EpgmPath) -> str:
    """Semantic label for the collapsed path, reported in the native trace.

    The contract-level edge in ``retrieved_subgraph`` is always ``feeds`` (the
    only provenance dependency type in ``ALLOWED_EDGE_TYPES``); this finer
    label is trace-only.
    """

    if len(path.steps) == 1:
        return path.steps[0].edge_type
    if any(
        step.edge_type == ProvenanceEdgeType.FEEDS.value for step in path.steps
    ):
        return ProvenanceEdgeType.FEEDS.value
    return ProvenanceEdgeType.DEPENDS_ON.value


def _logical_endpoints(path: EpgmPath) -> tuple[str, str]:
    """Orient the collapsed edge along the stored graph direction.

    A bidirectional walk may reach its target by going against the recorded
    orientation. Emitting ``seed -> target`` would then invert a dependency,
    so a mostly-reverse path is flipped before it becomes a public edge.
    """

    reverse_steps = sum(1 for step in path.steps if step.direction == "reverse")
    if reverse_steps * 2 > len(path.steps):
        return path.target_id, path.seed_id
    return path.seed_id, path.target_id


def _logical_edges(paths: tuple[EpgmPath, ...]) -> list[GraphEdge]:
    """Collapse each accepted path into one directed candidate-level edge."""

    edges: list[GraphEdge] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        source, target = _logical_endpoints(path)
        key = (source, target)
        if key in seen:
            continue
        seen.add(key)
        edges.append(
            GraphEdge(
                source=source,
                target=target,
                edge_type="feeds",
                weight=path.score,
                directed=True,
            )
        )
    return edges


def _edge_traces(
    paths: tuple[EpgmPath, ...],
    request: ExecutionProvenanceRankingRequest,
) -> tuple[ProvenanceEdgeTrace, ...]:
    edge_by_key = {
        (edge.source, edge.target, edge.edge_type.value): edge
        for edge in request.graph.edges
    }
    traversed = {
        (step.source, step.target, step.edge_type)
        for path in paths
        for step in path.steps
    }
    return tuple(
        _edge_trace(edge_by_key[key]) for key in sorted(traversed) if key in edge_by_key
    )


def _edge_trace(edge: ExecutionProvenanceEdge) -> ProvenanceEdgeTrace:
    semantic_rank = edge.metadata.get("semantic_rank")
    semantic_score = edge.metadata.get("semantic_score")
    return ProvenanceEdgeTrace(
        source=edge.source,
        target=edge.target,
        edge_type=edge.edge_type,
        weight=edge.weight,
        binding=(
            ProvenanceBindingTrace(
                output_field=edge.binding.output_field,
                input_parameter=edge.binding.input_parameter,
                binding_kind=edge.binding.binding_kind,
            )
            if edge.binding is not None
            else None
        ),
        semantic_rank=(
            cast(int, semantic_rank)
            if isinstance(semantic_rank, int) and not isinstance(semantic_rank, bool)
            else None
        ),
        semantic_score=(
            float(semantic_score)
            if isinstance(semantic_score, (int, float))
            and not isinstance(semantic_score, bool)
            else None
        ),
    )


def _scorer_identity(request: ExecutionProvenanceRankingRequest) -> str:
    identities = {
        value
        for edge in request.graph.edges
        if edge.edge_type is ProvenanceEdgeType.FEEDS
        if isinstance((value := edge.metadata.get("semantic_scorer")), str)
    }
    return ",".join(sorted(identities)) or "unknown"


__all__ = ["EpgmRankedNode", "EpgmRetrievalResult", "EpgmRetriever"]
