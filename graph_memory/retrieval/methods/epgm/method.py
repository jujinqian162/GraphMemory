from __future__ import annotations

from dataclasses import dataclass

from graph_memory.retrieval.contracts import RankedNode
from graph_memory.retrieval.methods.epgm.config import EpgmRetrieverConfig
from graph_memory.retrieval.methods.epgm.search import EpgmPath, search_epgm_paths
from graph_memory.retrieval.methods.flat.dense import DenseTaskRetriever
from graph_memory.retrieval.requests import (
    ExecutionProvenanceRankingRequest,
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
    """No-train EPGM provenance retriever (paper method: "ours").

    Dense seeds the ranking, then typed bidirectional graph propagation
    reranks candidates by the best evidentiary path that reaches them. Not
    registered in the retrieval registry: it is driven by the standalone EPGM
    runner, reusing the shared ExecutionProvenanceGraph / request contracts.
    """

    dense_ranker: DenseTaskRetriever
    config: EpgmRetrieverConfig = EpgmRetrieverConfig()
    name: str = "epgm_retriever"
    display_name: str = "ours"

    def rank(self, request: ExecutionProvenanceRankingRequest) -> EpgmRetrievalResult:
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
        # normalize dense scores into [0, 1] relevance for path scoring
        seed_relevance = _normalized_relevance(dense_ranked)
        candidate_ids = frozenset(dense_score)
        seed_ids = tuple(node.node_id for node in dense_ranked[: self.config.seed_top_s])

        paths = search_epgm_paths(
            request.graph,
            seed_ids,
            seed_relevance,
            candidate_ids=candidate_ids,
            config=self.config,
        )
        best_path_by_target: dict[str, EpgmPath] = {}
        for path in paths:
            current = best_path_by_target.get(path.target_id)
            if current is None or path.score > current.score:
                best_path_by_target[path.target_id] = path

        ranked: list[EpgmRankedNode] = []
        for node in dense_ranked:
            base = seed_relevance.get(node.node_id, 0.0)
            best = best_path_by_target.get(node.node_id)
            graph_score = best.score if best is not None else 0.0
            # Additive fusion: dense relevance is a floor that graph propagation
            # can only lift, never demote. A strong dense hit that no path
            # reaches (e.g. a seed, or a direct answer node) keeps its base
            # score instead of being averaged down. This preserves easy
            # factual questions while still promoting graph-connected evidence
            # for multi-hop / provenance questions.
            fused = base + self.config.graph_weight * graph_score
            ranked.append(
                EpgmRankedNode(
                    node_id=node.node_id,
                    score=fused,
                    dense_rank=dense_rank[node.node_id],
                    dense_score=dense_score[node.node_id],
                    graph_score=graph_score,
                    best_path=best,
                )
            )
        ranked.sort(key=lambda item: (-item.score, item.dense_rank, item.node_id))
        return EpgmRetrievalResult(
            ranked_nodes=tuple(ranked),
            seed_ids=seed_ids,
            paths=paths,
        )


def _normalized_relevance(dense_ranked: list[RankedNode]) -> dict[str, float]:
    if not dense_ranked:
        return {}
    scores = [node.score for node in dense_ranked]
    lo = min(scores)
    hi = max(scores)
    span = hi - lo
    if span <= 0.0:
        return {node.node_id: 1.0 for node in dense_ranked}
    return {node.node_id: (node.score - lo) / span for node in dense_ranked}


__all__ = ["EpgmRankedNode", "EpgmRetrievalResult", "EpgmRetriever"]
