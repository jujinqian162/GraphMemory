from __future__ import annotations


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    min_score = min(scores.values())
    max_score = max(scores.values())
    if max_score == min_score:
        return {node_id: 0.0 for node_id in scores}
    return {node_id: (score - min_score) / (max_score - min_score) for node_id, score in scores.items()}


def normalize_component_scores(scores: dict[str, float], node_ids: set[str]) -> dict[str, float]:
    zero_filled_scores = {node_id: scores.get(node_id, 0.0) for node_id in node_ids}
    return normalize_scores(zero_filled_scores)
