from __future__ import annotations

from collections.abc import Mapping, Sequence

from graph_memory.retrieval.methods.fast_graphrag.config import FastGraphRAGScoringConfig
from graph_memory.retrieval.requests import FastGraphRAGKnowledgeGraph, TextCandidate


def score_relations(
    kg: FastGraphRAGKnowledgeGraph,
    entity_scores: Mapping[str, float],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for relation in kg.relations:
        source_score = float(entity_scores.get(relation.source_entity_id, 0.0))
        target_score = float(entity_scores.get(relation.target_entity_id, 0.0))
        scores[relation.relation_id] = ((source_score + target_score) / 2.0) * relation.weight
    return scores


def score_candidates(
    candidates: Sequence[TextCandidate],
    kg: FastGraphRAGKnowledgeGraph,
    *,
    entity_scores: Mapping[str, float],
    dense_fallback_scores: Mapping[str, float],
    config: FastGraphRAGScoringConfig,
) -> dict[str, float]:
    relation_scores = score_relations(kg, entity_scores)
    scores = {
        candidate.item_id: config.lambda_dense_fallback * float(dense_fallback_scores.get(candidate.item_id, 0.0))
        for candidate in candidates
    }
    entity_contributions: dict[str, list[float]] = {candidate.item_id: [] for candidate in candidates}
    for entity in kg.entities:
        contribution = float(entity_scores.get(entity.entity_id, 0.0))
        for candidate_id in entity.candidate_ids:
            entity_contributions.setdefault(candidate_id, []).append(contribution)
    for candidate_id, contributions in entity_contributions.items():
        scores[candidate_id] = scores.get(candidate_id, 0.0) + config.lambda_entity * _average(contributions)

    relation_contributions: dict[str, list[float]] = {candidate.item_id: [] for candidate in candidates}
    for relation in kg.relations:
        contribution = relation_scores.get(relation.relation_id, 0.0)
        for candidate_id in relation.candidate_ids:
            relation_contributions.setdefault(candidate_id, []).append(contribution)
    for candidate_id, contributions in relation_contributions.items():
        scores[candidate_id] = scores.get(candidate_id, 0.0) + config.lambda_relation * _average(contributions)
    return scores


def _average(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


__all__ = ["FastGraphRAGScoringConfig", "score_candidates", "score_relations"]
