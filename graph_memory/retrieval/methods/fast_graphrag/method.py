from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from graph_memory.graphs.views import induced_retrieved_subgraph
from graph_memory.retrieval.contracts import RankedNode, RetrievalMethodResult, RetrievalTrace
from graph_memory.retrieval.methods.flat.dense import DenseTaskRetriever
from graph_memory.retrieval.methods.fast_graphrag.nlp import normalize_entity_text
from graph_memory.retrieval.methods.fast_graphrag.pagerank import personalized_pagerank
from graph_memory.retrieval.methods.fast_graphrag.scoring import FastGraphRAGScoringConfig, score_candidates
from graph_memory.retrieval.requests import (
    DenseConfigLike,
    FastGraphRAGEntity,
    FastGraphRAGRequest,
    RankingMethodRequest,
    TextCandidate,
    TextRankingRequest,
)

if TYPE_CHECKING:
    from graph_memory.embeddings import SentenceEncoder


class FastGraphRAGDenseScorer(Protocol):
    def score_entities(self, query_text: str, entities: Sequence[FastGraphRAGEntity]) -> Mapping[str, float]:
        ...

    def score_candidates(self, query_text: str, candidates: Sequence[TextCandidate]) -> Mapping[str, float]:
        ...


class DenseFastGraphRAGScorer:
    def __init__(self, *, config: DenseConfigLike, encoder: "SentenceEncoder | None" = None) -> None:
        self._ranker = DenseTaskRetriever(config=config, encoder=encoder)

    def score_entities(self, query_text: str, entities: Sequence[FastGraphRAGEntity]) -> Mapping[str, float]:
        request = TextRankingRequest(
            task_id="fast_graphrag_entities",
            query_text=query_text,
            candidates=tuple(
                TextCandidate(
                    item_id=entity.entity_id,
                    text=f"{entity.name}\n{entity.description}",
                    metadata={},
                )
                for entity in entities
            ),
        )
        return {ranked_node.node_id: ranked_node.score for ranked_node in self._ranker.rank(request)}

    def score_candidates(self, query_text: str, candidates: Sequence[TextCandidate]) -> Mapping[str, float]:
        request = TextRankingRequest(
            task_id="fast_graphrag_candidates",
            query_text=query_text,
            candidates=candidates,
        )
        return {ranked_node.node_id: ranked_node.score for ranked_node in self._ranker.rank(request)}


@dataclass(frozen=True)
class FastGraphRAGConfig:
    ppr_damping: float = 0.85
    ppr_max_iterations: int = 100
    ppr_tolerance: float = 1e-8
    lexical_exact_match_score: float = 1.0
    lexical_substring_match_score: float = 0.5
    scoring: FastGraphRAGScoringConfig = field(default_factory=FastGraphRAGScoringConfig)


@dataclass(frozen=True)
class FastGraphRAGMethod:
    name: str
    config: FastGraphRAGConfig
    dense_ranker: FastGraphRAGDenseScorer

    def rank_task(self, request: RankingMethodRequest, *, top_k: int) -> RetrievalMethodResult:
        if not isinstance(request, FastGraphRAGRequest):
            raise TypeError(f"{self.name} requires FastGraphRAGRequest, got {type(request).__name__}.")

        dense_entity_scores = self.dense_ranker.score_entities(request.query_text, request.knowledge_graph.entities)
        lexical_scores = _lexical_entity_seed_scores(request, self.config)
        seed_scores = _merge_seed_scores(lexical_scores, dense_entity_scores)
        entity_scores = personalized_pagerank(
            _entity_adjacency(request),
            seed_scores,
            damping=self.config.ppr_damping,
            max_iterations=self.config.ppr_max_iterations,
            tolerance=self.config.ppr_tolerance,
        )
        candidate_scores = score_candidates(
            request.candidates,
            request.knowledge_graph,
            entity_scores=entity_scores,
            dense_fallback_scores=self.dense_ranker.score_candidates(request.query_text, request.candidates),
            config=self.config.scoring,
        )
        ranked_nodes = _rank_candidates(request.candidates, candidate_scores)
        top_node_ids = [ranked_node.node_id for ranked_node in ranked_nodes[:top_k]]
        retrieved_subgraph = induced_retrieved_subgraph(request.candidate_graph, top_node_ids)
        return RetrievalMethodResult(
            ranked_nodes=ranked_nodes,
            trace=RetrievalTrace(retrieved_edges=retrieved_subgraph["edges"]),
        )


def _entity_adjacency(request: FastGraphRAGRequest) -> dict[str, dict[str, float]]:
    adjacency = {
        entity.entity_id: {}
        for entity in request.knowledge_graph.entities
    }
    for relation in request.knowledge_graph.relations:
        _add_weight(adjacency, relation.source_entity_id, relation.target_entity_id, relation.weight)
        _add_weight(adjacency, relation.target_entity_id, relation.source_entity_id, relation.weight)
    return adjacency


def _add_weight(adjacency: dict[str, dict[str, float]], source_id: str, target_id: str, weight: float) -> None:
    adjacency.setdefault(source_id, {})
    adjacency.setdefault(target_id, {})
    adjacency[source_id][target_id] = adjacency[source_id].get(target_id, 0.0) + weight


def _lexical_entity_seed_scores(
    request: FastGraphRAGRequest,
    config: FastGraphRAGConfig,
) -> dict[str, float]:
    query_norm = normalize_entity_text(request.query_text)
    scores: dict[str, float] = {}
    for entity in request.knowledge_graph.entities:
        entity_score = 0.0
        for alias in _entity_aliases(entity):
            if query_norm == alias:
                entity_score = max(entity_score, config.lexical_exact_match_score)
            elif _contains_words(query_norm, alias):
                entity_score = max(entity_score, config.lexical_substring_match_score)
        if entity_score > 0.0:
            scores[entity.entity_id] = entity_score
    return scores


def _entity_aliases(entity: FastGraphRAGEntity) -> tuple[str, ...]:
    aliases = {entity.normalized_name, normalize_entity_text(entity.name)}
    aliases.update(normalize_entity_text(line) for line in entity.description.splitlines())
    return tuple(alias for alias in sorted(aliases) if alias)


def _contains_words(text: str, needle: str) -> bool:
    if not needle:
        return False
    return re.search(rf"(?:^|\s){re.escape(needle)}(?:\s|$)", text) is not None


def _merge_seed_scores(
    lexical_scores: Mapping[str, float],
    dense_scores: Mapping[str, float],
) -> dict[str, float]:
    entity_ids = set(lexical_scores) | set(dense_scores)
    return {
        entity_id: max(float(lexical_scores.get(entity_id, 0.0)), float(dense_scores.get(entity_id, 0.0)))
        for entity_id in entity_ids
    }


def _rank_candidates(
    candidates: Sequence[TextCandidate],
    scores: Mapping[str, float],
) -> list[RankedNode]:
    ranked_nodes = [
        RankedNode(node_id=candidate.item_id, score=float(scores.get(candidate.item_id, 0.0)))
        for candidate in candidates
    ]
    return sorted(ranked_nodes, key=lambda node: (-node.score, _candidate_position(candidates, node.node_id), node.node_id))


def _candidate_position(candidates: Sequence[TextCandidate], node_id: str) -> float:
    for candidate in candidates:
        if candidate.item_id != node_id:
            continue
        position = candidate.metadata.get("position")
        if isinstance(position, bool):
            return float("inf")
        if isinstance(position, int | float):
            return float(position)
    return float("inf")


__all__ = [
    "DenseFastGraphRAGScorer",
    "FastGraphRAGConfig",
    "FastGraphRAGDenseScorer",
    "FastGraphRAGMethod",
]
