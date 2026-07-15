from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from graph_memory.retrieval.contracts import (
    EntityRelationTrace,
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
    EntityKnowledgeGraph,
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
        _ = top_k
        if not isinstance(request, GraphRAGRequest):
            raise TypeError(
                f"{self.name} requires GraphRAGRequest, got {type(request).__name__}."
            )
        candidate_dense_scores = _dense_candidate_scores(self.dense_ranker, request)
        entity_dense_scores = _dense_entity_scores(self.dense_ranker, request)
        linked_entity_ids = link_query_entities(
            request.query_text, request.knowledge_graph
        )
        lexical_scores = lexical_entity_scores(
            request.query_text, request.knowledge_graph
        )
        seed_scores = _merge_seed_scores(
            request.knowledge_graph,
            linked_entity_ids=linked_entity_ids,
            lexical_scores=lexical_scores,
            dense_scores=entity_dense_scores,
            seed_top_s=self.config.seed_top_s,
        )
        entity_scores = _personalized_pagerank(
            request.knowledge_graph,
            seed_scores,
            self.config,
        )
        graph_scores = _project_graph_scores(request.knowledge_graph, entity_scores)
        total_weight = self.config.semantic_weight + self.config.entity_weight
        ranked_nodes = [
            RankedNode(
                candidate.item_id,
                (
                    self.config.semantic_weight
                    * candidate_dense_scores.get(candidate.item_id, 0.0)
                    + self.config.entity_weight
                    * graph_scores.get(candidate.item_id, 0.0)
                )
                / total_weight,
            )
            for candidate in request.candidates
        ]
        ranked_nodes.sort(key=lambda node: (-node.score, node.node_id))
        traced_relations = tuple(
            EntityRelationTrace(
                source_entity_id=relation.source_entity_id,
                target_entity_id=relation.target_entity_id,
                weight=relation.weight,
                candidate_ids=relation.candidate_ids,
            )
            for relation in request.knowledge_graph.relations
            if entity_scores.get(relation.source_entity_id, 0.0) > 0.0
            or entity_scores.get(relation.target_entity_id, 0.0) > 0.0
        )
        return RetrievalMethodResult(
            ranked_nodes=ranked_nodes,
            trace=RetrievalTrace(
                native_trace=GraphRAGTrace(
                    entity_ids=tuple(
                        entity.entity_id for entity in request.knowledge_graph.entities
                    ),
                    linked_entity_ids=linked_entity_ids,
                    seed_entity_ids=tuple(sorted(seed_scores)),
                    relations=traced_relations,
                )
            ),
        )


def _dense_candidate_scores(
    ranker: DenseTaskRetriever,
    request: GraphRAGRequest,
) -> dict[str, float]:
    ranked = ranker.rank(
        TextRankingRequest(request.task_id, request.query_text, request.candidates)
    )
    return {node.node_id: node.score for node in ranked}


def _dense_entity_scores(
    ranker: DenseTaskRetriever,
    request: GraphRAGRequest,
) -> dict[str, float]:
    graph = request.knowledge_graph
    if not graph.entities:
        return {}
    ranked = ranker.rank(
        TextRankingRequest(
            task_id=f"{request.task_id}:entities",
            query_text=request.query_text,
            candidates=tuple(
                TextCandidate(
                    entity.entity_id,
                    f"{entity.name}\n{entity.description}",
                    {},
                )
                for entity in graph.entities
            ),
        )
    )
    return {node.node_id: node.score for node in ranked}


def _merge_seed_scores(
    graph: EntityKnowledgeGraph,
    *,
    linked_entity_ids: tuple[str, ...],
    lexical_scores: dict[str, float],
    dense_scores: dict[str, float],
    seed_top_s: int,
) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for entity_id in linked_entity_ids:
        scores[entity_id] += 1.0
    for entity_id, score in lexical_scores.items():
        scores[entity_id] += 0.5 * max(score, 0.0)
    dense_seed_ids = sorted(
        (entity.entity_id for entity in graph.entities),
        key=lambda entity_id: (-dense_scores.get(entity_id, 0.0), entity_id),
    )[: min(seed_top_s, len(graph.entities))]
    for entity_id in dense_seed_ids:
        scores[entity_id] += max(dense_scores.get(entity_id, 0.0), 0.0)
    if not scores and graph.entities:
        uniform = 1.0 / len(graph.entities)
        return {entity.entity_id: uniform for entity in graph.entities}
    return dict(scores)


def _personalized_pagerank(
    graph: EntityKnowledgeGraph,
    seed_scores: dict[str, float],
    config: GraphRAGConfig,
) -> dict[str, float]:
    entity_ids = tuple(entity.entity_id for entity in graph.entities)
    if not entity_ids:
        return {}
    positive = {entity_id: max(seed_scores.get(entity_id, 0.0), 0.0) for entity_id in entity_ids}
    total = sum(positive.values())
    personalization = (
        {entity_id: score / total for entity_id, score in positive.items()}
        if total > 0.0
        else {entity_id: 1.0 / len(entity_ids) for entity_id in entity_ids}
    )
    adjacency: dict[str, dict[str, float]] = {entity_id: {} for entity_id in entity_ids}
    for relation in graph.relations:
        adjacency[relation.source_entity_id][relation.target_entity_id] = relation.weight
        adjacency[relation.target_entity_id][relation.source_entity_id] = relation.weight
    scores = dict(personalization)
    for _ in range(config.max_iterations):
        propagated = {entity_id: 0.0 for entity_id in entity_ids}
        dangling = 0.0
        for source, source_score in scores.items():
            neighbors = adjacency[source]
            neighbor_total = sum(neighbors.values())
            if neighbor_total <= 0.0:
                dangling += source_score
                continue
            for target, weight in neighbors.items():
                propagated[target] += source_score * weight / neighbor_total
        updated = {
            entity_id: config.restart_probability * personalization[entity_id]
            + (1.0 - config.restart_probability)
            * (
                propagated[entity_id]
                + dangling * personalization[entity_id]
            )
            for entity_id in entity_ids
        }
        if sum(abs(updated[key] - scores[key]) for key in entity_ids) <= config.convergence_tolerance:
            scores = updated
            break
        scores = updated
    return scores


def _project_graph_scores(
    graph: EntityKnowledgeGraph,
    entity_scores: dict[str, float],
) -> dict[str, float]:
    entity_contributions: dict[str, list[float]] = defaultdict(list)
    for entity in graph.entities:
        for candidate_id in entity.candidate_ids:
            entity_contributions[candidate_id].append(
                entity_scores.get(entity.entity_id, 0.0)
            )
    relation_contributions: dict[str, list[float]] = defaultdict(list)
    for relation in graph.relations:
        relation_score = relation.weight * (
            entity_scores.get(relation.source_entity_id, 0.0)
            + entity_scores.get(relation.target_entity_id, 0.0)
        ) / 2.0
        for candidate_id in relation.candidate_ids:
            relation_contributions[candidate_id].append(relation_score)
    candidate_ids = set(entity_contributions) | set(relation_contributions)
    return {
        candidate_id: _mean(entity_contributions.get(candidate_id, []))
        + _mean(relation_contributions.get(candidate_id, []))
        for candidate_id in candidate_ids
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


__all__ = ["GraphRAGMethod"]
