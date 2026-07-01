from __future__ import annotations

import math

from graph_memory.retrieval.requests import TemporalMemoryRankingRequest


def request_importance_scores(
    request: TemporalMemoryRankingRequest,
    *,
    require_complete: bool,
) -> dict[str, float]:
    candidate_ids = {candidate.item_id for candidate in request.candidates}
    observed_ids = set(request.importance_by_item_id)
    extra = sorted(observed_ids - candidate_ids)
    if extra:
        raise ValueError(
            f"Memory Stream request task_id={request.task_id} has importance for unknown items: {extra}."
        )
    missing = sorted(candidate_ids - observed_ids)
    if require_complete and missing:
        raise ValueError(
            "Memory Stream request "
            f"task_id={request.task_id} missing importance scores for items: {missing}."
        )

    scores: dict[str, float] = {}
    for item_id in sorted(candidate_ids):
        raw_score = request.importance_by_item_id.get(item_id, 0.0)
        if isinstance(raw_score, bool):
            raise ValueError(
                f"Memory Stream request task_id={request.task_id} item_id={item_id} importance must be numeric."
            )
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Memory Stream request task_id={request.task_id} item_id={item_id} importance must be numeric."
            ) from error
        if not math.isfinite(score):
            raise ValueError(
                f"Memory Stream request task_id={request.task_id} item_id={item_id} importance must be finite."
            )
        scores[item_id] = score
    return scores


__all__ = ["request_importance_scores"]
