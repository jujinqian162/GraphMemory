from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from graph_memory.retrieval.contracts import (
    GraphRAGCandidateScoreTrace,
    GraphRAGEntityScoreTrace,
    GraphRAGTrace,
    RankedNode,
    RetrievalMethodResult,
    RetrievalTrace,
)
from graph_memory.retrieval.methods.flat.dense import DenseTaskRetriever
from graph_memory.retrieval.methods.graphrag.config import GraphRAGConfig
from graph_memory.retrieval.methods.graphrag.index import (
    lexical_entity_scores,
    link_query_entities,
)
from graph_memory.retrieval.requests import (
    GraphRAGKnowledgeGraph,
    GraphRAGRequest,
    RankingMethodRequest,
    TextCandidate,
    TextRankingRequest,
)


@dataclass(frozen=True)
class GraphRAGMethod:
    dense_ranker: DenseTaskRetriever
    config: GraphRAGConfig = GraphRAGConfig()
    name: str = "graphrag"

    def rank_task(
        self,
        request: RankingMethodRequest,
        *,
        top_k: int,
    ) -> RetrievalMethodResult:
        del top_k
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
        dense_scores = {node.node_id: node.score for node in dense_ranked}
        graph = request.knowledge_graph
        if not graph.entities or not graph.relations:
            return _dense_fallback(
                request,
                dense_ranked,
                graph,
                trace_top_entities=self.config.trace_top_entities,
            )

        entity_semantic_scores = _entity_semantic_scores(self.dense_ranker, request)
        linked_entity_ids = link_query_entities(request.query_text, graph)
        seed_scores = _seed_scores(
            graph,
            linked_entity_ids=linked_entity_ids,
            lexical_scores=lexical_entity_scores(request.query_text, graph),
            semantic_scores=entity_semantic_scores,
            config=self.config,
        )
        entity_scores, iterations, converged = _personalized_pagerank(
            graph,
            seed_scores=seed_scores,
            config=self.config,
        )
        projected_scores = _project_entity_scores(graph, entity_scores)
        if not projected_scores or max(projected_scores.values(), default=0.0) <= 0.0:
            return _dense_fallback(
                request,
                dense_ranked,
                graph,
                linked_entity_ids=linked_entity_ids,
                seed_scores=seed_scores,
                entity_scores=entity_scores,
                iterations=iterations,
                converged=converged,
                trace_top_entities=self.config.trace_top_entities,
            )

        normalized_dense = _minmax_scores(dense_scores)
        normalized_graph = _minmax_scores(
            {
                candidate.item_id: projected_scores.get(candidate.item_id, 0.0)
                for candidate in request.candidates
            }
        )
        total_weight = self.config.semantic_weight + self.config.graph_weight
        final_scores = {
            candidate.item_id: (
                self.config.semantic_weight * normalized_dense[candidate.item_id]
                + self.config.graph_weight * normalized_graph[candidate.item_id]
            )
            / total_weight
            for candidate in request.candidates
        }
        ranked_nodes = tuple(
            RankedNode(node_id=candidate_id, score=score)
            for candidate_id, score in sorted(
                final_scores.items(), key=lambda item: (-item[1], item[0])
            )
        )
        trace = _trace(
            dense_ranked=dense_ranked,
            ranked_nodes=ranked_nodes,
            graph=graph,
            linked_entity_ids=linked_entity_ids,
            seed_scores=seed_scores,
            entity_scores=entity_scores,
            projected_scores=normalized_graph,
            iterations=iterations,
            converged=converged,
            exact_dense_fallback=False,
            trace_top_entities=self.config.trace_top_entities,
        )
        return RetrievalMethodResult(
            ranked_nodes=ranked_nodes,
            trace=RetrievalTrace(native_trace=trace),
        )


def _entity_semantic_scores(
    ranker: DenseTaskRetriever,
    request: GraphRAGRequest,
) -> dict[str, float]:
    entities = request.knowledge_graph.entities
    if not entities:
        return {}
    ranked = ranker.rank(
        TextRankingRequest(
            task_id=f"{request.task_id}:graphrag-entities",
            query_text=request.query_text,
            candidates=tuple(
                TextCandidate(
                    item_id=entity.entity_id,
                    text=entity.name,
                    metadata={"representation": "fast_graphrag_entity"},
                )
                for entity in entities
            ),
        )
    )
    return {node.node_id: node.score for node in ranked}


def _seed_scores(
    graph: GraphRAGKnowledgeGraph,
    *,
    linked_entity_ids: tuple[str, ...],
    lexical_scores: dict[str, float],
    semantic_scores: dict[str, float],
    config: GraphRAGConfig,
) -> dict[str, float]:
    result: dict[str, float] = defaultdict(float)
    for entity_id in linked_entity_ids:
        result[entity_id] += config.exact_match_weight
    lexical_seed_ids = sorted(
        lexical_scores,
        key=lambda entity_id: (-lexical_scores[entity_id], entity_id),
    )[: min(config.seed_top_s, len(lexical_scores))]
    for entity_id in lexical_seed_ids:
        result[entity_id] += config.lexical_weight * max(lexical_scores[entity_id], 0.0)
    semantic_seed_ids = sorted(
        (entity.entity_id for entity in graph.entities),
        key=lambda entity_id: (-semantic_scores.get(entity_id, 0.0), entity_id),
    )[: min(config.seed_top_s, len(graph.entities))]
    normalized_semantic = _minmax_scores(
        {
            entity_id: semantic_scores.get(entity_id, 0.0)
            for entity_id in semantic_seed_ids
        }
    )
    for entity_id in semantic_seed_ids:
        result[entity_id] += (
            config.semantic_seed_weight * normalized_semantic[entity_id]
        )
    positive = {entity_id: score for entity_id, score in result.items() if score > 0.0}
    if positive:
        return positive
    if not graph.entities:
        return {}
    uniform = 1.0 / len(graph.entities)
    return {entity.entity_id: uniform for entity in graph.entities}


def _personalized_pagerank(
    graph: GraphRAGKnowledgeGraph,
    *,
    seed_scores: dict[str, float],
    config: GraphRAGConfig,
) -> tuple[dict[str, float], int, bool]:
    entity_ids = tuple(entity.entity_id for entity in graph.entities)
    if not entity_ids:
        return {}, 0, True
    total_seed = sum(
        max(seed_scores.get(entity_id, 0.0), 0.0) for entity_id in entity_ids
    )
    personalization = (
        {
            entity_id: max(seed_scores.get(entity_id, 0.0), 0.0) / total_seed
            for entity_id in entity_ids
        }
        if total_seed > 0.0
        else {entity_id: 1.0 / len(entity_ids) for entity_id in entity_ids}
    )
    adjacency: dict[str, dict[str, float]] = {entity_id: {} for entity_id in entity_ids}
    for relation in graph.relations:
        adjacency[relation.source_entity_id][relation.target_entity_id] = (
            relation.weight
        )
        adjacency[relation.target_entity_id][relation.source_entity_id] = (
            relation.weight
        )

    scores = dict(personalization)
    for iteration in range(1, config.max_iterations + 1):
        propagated = {entity_id: 0.0 for entity_id in entity_ids}
        dangling = 0.0
        for source, source_score in scores.items():
            neighbors = adjacency[source]
            weight_sum = sum(neighbors.values())
            if weight_sum <= 0.0:
                dangling += source_score
                continue
            for target, weight in neighbors.items():
                propagated[target] += source_score * weight / weight_sum
        updated = {
            entity_id: config.restart_probability * personalization[entity_id]
            + (1.0 - config.restart_probability)
            * (propagated[entity_id] + dangling * personalization[entity_id])
            for entity_id in entity_ids
        }
        delta = sum(abs(updated[key] - scores[key]) for key in entity_ids)
        scores = updated
        if delta <= len(entity_ids) * config.convergence_tolerance:
            return scores, iteration, True
    return scores, config.max_iterations, False


def _project_entity_scores(
    graph: GraphRAGKnowledgeGraph,
    entity_scores: dict[str, float],
) -> dict[str, float]:
    scores_by_unit: dict[str, float] = defaultdict(float)
    for entity in graph.entities:
        score = entity_scores.get(entity.entity_id, 0.0)
        for unit_id in entity.text_unit_ids:
            scores_by_unit[unit_id] += score
    candidate_scores: dict[str, float] = defaultdict(float)
    for unit in graph.text_units:
        unit_score = scores_by_unit.get(unit.unit_id, 0.0)
        for candidate_id in unit.candidate_ids:
            candidate_scores[candidate_id] = max(
                candidate_scores[candidate_id], unit_score
            )
    return dict(candidate_scores)


def _minmax_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    minimum = min(scores.values())
    maximum = max(scores.values())
    if maximum <= minimum:
        return {key: 1.0 if maximum > 0.0 else 0.0 for key in scores}
    scale = maximum - minimum
    return {key: (value - minimum) / scale for key, value in scores.items()}


def _dense_fallback(
    request: GraphRAGRequest,
    dense_ranked: list[RankedNode],
    graph: GraphRAGKnowledgeGraph,
    *,
    linked_entity_ids: tuple[str, ...] = (),
    seed_scores: dict[str, float] | None = None,
    entity_scores: dict[str, float] | None = None,
    iterations: int = 0,
    converged: bool = True,
    trace_top_entities: int,
) -> RetrievalMethodResult:
    ranked_nodes = tuple(dense_ranked)
    return RetrievalMethodResult(
        ranked_nodes=ranked_nodes,
        trace=RetrievalTrace(
            native_trace=_trace(
                dense_ranked=dense_ranked,
                ranked_nodes=ranked_nodes,
                graph=graph,
                linked_entity_ids=linked_entity_ids,
                seed_scores=seed_scores or {},
                entity_scores=entity_scores or {},
                projected_scores={
                    candidate.item_id: 0.0 for candidate in request.candidates
                },
                iterations=iterations,
                converged=converged,
                exact_dense_fallback=True,
                trace_top_entities=trace_top_entities,
            )
        ),
    )


def _trace(
    *,
    dense_ranked: list[RankedNode],
    ranked_nodes: tuple[RankedNode, ...],
    graph: GraphRAGKnowledgeGraph,
    linked_entity_ids: tuple[str, ...],
    seed_scores: dict[str, float],
    entity_scores: dict[str, float],
    projected_scores: dict[str, float],
    iterations: int,
    converged: bool,
    exact_dense_fallback: bool,
    trace_top_entities: int,
) -> GraphRAGTrace:
    dense_rank = {node.node_id: rank for rank, node in enumerate(dense_ranked, 1)}
    final_rank = {node.node_id: rank for rank, node in enumerate(ranked_nodes, 1)}
    dense_score = {node.node_id: node.score for node in dense_ranked}
    final_score = {node.node_id: node.score for node in ranked_nodes}
    top_entities = sorted(entity_scores.items(), key=lambda item: (-item[1], item[0]))[
        :trace_top_entities
    ]
    entity_name = {entity.entity_id: entity.name for entity in graph.entities}
    return GraphRAGTrace(
        text_unit_count=len(graph.text_units),
        entity_count=len(graph.entities),
        relation_count=len(graph.relations),
        linked_entity_ids=linked_entity_ids,
        seed_entity_scores=tuple(
            GraphRAGEntityScoreTrace(
                entity_id=entity_id,
                name=entity_name[entity_id],
                score=score,
            )
            for entity_id, score in sorted(seed_scores.items())
        ),
        top_entity_scores=tuple(
            GraphRAGEntityScoreTrace(
                entity_id=entity_id,
                name=entity_name[entity_id],
                score=score,
            )
            for entity_id, score in top_entities
        ),
        candidate_scores=tuple(
            GraphRAGCandidateScoreTrace(
                candidate_id=node.node_id,
                dense_score=dense_score[node.node_id],
                graph_score=projected_scores.get(node.node_id, 0.0),
                final_score=final_score[node.node_id],
                dense_rank=dense_rank[node.node_id],
                final_rank=final_rank[node.node_id],
            )
            for node in dense_ranked
        ),
        iterations=iterations,
        converged=converged,
        exact_dense_fallback=exact_dense_fallback,
    )


__all__ = ["GraphRAGMethod"]
