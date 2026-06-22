from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from graph_memory.retrieval.requests import FastGraphRAGKnowledgeGraph, TextCandidate


@dataclass(frozen=True)
class FastGraphRAGScoringConfig:
    lambda_entity: float = 1.0
    lambda_relation: float = 1.0
    lambda_dense_fallback: float = 0.05


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
    for entity in kg.entities:
        contribution = config.lambda_entity * float(entity_scores.get(entity.entity_id, 0.0))
        for candidate_id in entity.candidate_ids:
            scores[candidate_id] = scores.get(candidate_id, 0.0) + contribution
    for relation in kg.relations:
        contribution = config.lambda_relation * relation_scores.get(relation.relation_id, 0.0)
        for candidate_id in relation.candidate_ids:
            scores[candidate_id] = scores.get(candidate_id, 0.0) + contribution
    return scores


__all__ = ["FastGraphRAGScoringConfig", "score_candidates", "score_relations"]
