from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.retrieval.contracts import RankedNode, SeedRanker

DEFAULT_BULK_TASK_GROUP_SIZE = 32


@runtime_checkable
class BulkSeedRanker(Protocol):
    def rank_many(self, task_inputs: list[MemoryTaskInput]) -> list[list[RankedNode]]:
        ...


def rank_tasks(
    ranker: SeedRanker,
    task_inputs: Sequence[MemoryTaskInput],
) -> list[list[RankedNode]]:
    task_list = list(task_inputs)
    if isinstance(ranker, BulkSeedRanker):
        results = ranker.rank_many(task_list)
        if len(results) != len(task_list):
            raise ValueError(
                "Bulk seed ranker returned an invalid result count: "
                f"expected={len(task_list)} observed={len(results)}."
            )
        return results
    return [ranker.rank(task_input) for task_input in task_list]


def task_groups(
    task_inputs: Sequence[MemoryTaskInput],
    *,
    group_size: int = DEFAULT_BULK_TASK_GROUP_SIZE,
) -> list[list[MemoryTaskInput]]:
    if group_size <= 0:
        raise ValueError("Bulk task group_size must be positive.")
    task_list = list(task_inputs)
    return [
        task_list[start : start + group_size]
        for start in range(0, len(task_list), group_size)
    ]


__all__ = [
    "BulkSeedRanker",
    "DEFAULT_BULK_TASK_GROUP_SIZE",
    "rank_tasks",
    "task_groups",
]
