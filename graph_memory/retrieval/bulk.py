from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from graph_memory.retrieval.contracts import RankedNode, SeedRanker
from graph_memory.retrieval.requests import TextRankingRequest

DEFAULT_BULK_TASK_GROUP_SIZE = 32


@runtime_checkable
class BulkSeedRanker(Protocol):
    def rank_many(self, requests: list[TextRankingRequest]) -> list[list[RankedNode]]:
        ...


def rank_tasks(
    ranker: SeedRanker,
    requests: Sequence[TextRankingRequest],
) -> list[list[RankedNode]]:
    request_list = list(requests)
    if isinstance(ranker, BulkSeedRanker):
        results = ranker.rank_many(request_list)
        if len(results) != len(request_list):
            raise ValueError(
                "Bulk seed ranker returned an invalid result count: "
                f"expected={len(request_list)} observed={len(results)}."
            )
        return results
    return [ranker.rank(request) for request in request_list]


def task_groups(
    requests: Sequence[TextRankingRequest],
    *,
    group_size: int = DEFAULT_BULK_TASK_GROUP_SIZE,
) -> list[list[TextRankingRequest]]:
    if group_size <= 0:
        raise ValueError("Bulk task group_size must be positive.")
    request_list = list(requests)
    return [
        request_list[start : start + group_size]
        for start in range(0, len(request_list), group_size)
    ]


__all__ = [
    "BulkSeedRanker",
    "DEFAULT_BULK_TASK_GROUP_SIZE",
    "rank_tasks",
    "task_groups",
]
