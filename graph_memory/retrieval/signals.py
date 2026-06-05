from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from graph_memory.contracts.common import NodeId
from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.retrieval.contracts import Retriever


@dataclass(frozen=True)
class SeedSignal:
    """
    Frozen seed retrieval signal for one memory node.
    一个 memory node 的冻结初始检索信号。
    """

    node_id: NodeId
    score: float
    rank: int
    rank_percentile: float


class SeedSignalProvider(Protocol):
    """
    Replaceable provider for frozen seed retrieval signals.
    可替换的冻结初始检索信号提供器。
    """

    def score_task(self, task_input: MemoryTaskInput) -> list[SeedSignal]:
        ...


@dataclass(frozen=True)
class RetrieverSeedSignalProvider:
    """
    Seed signal provider backed by an existing flat retriever.
    基于现有 flat retriever 的 seed signal provider。
    """

    retriever: Retriever

    def score_task(self, task_input: MemoryTaskInput) -> list[SeedSignal]:
        ranked_nodes = self.retriever.rank(task_input) # HUMAN REVIEW POINT: 为什么不是用RetrievalMethod这个协议？
        expected_node_ids = {memory_item["id"] for memory_item in task_input["memory_items"]}
        observed_node_ids = {ranked_node.node_id for ranked_node in ranked_nodes}
        if observed_node_ids != expected_node_ids:
            missing = sorted(expected_node_ids - observed_node_ids)
            extra = sorted(observed_node_ids - expected_node_ids)
            raise ValueError(f"Seed retriever must return every memory node exactly once; missing={missing} extra={extra}.")

        sorted_nodes = sorted(ranked_nodes, key=lambda ranked_node: (-ranked_node.score, ranked_node.node_id))
        denominator = max(1, len(sorted_nodes) - 1)
        return [
            SeedSignal(
                node_id=ranked_node.node_id,
                score=float(ranked_node.score),
                rank=rank,
                rank_percentile=0.0 if len(sorted_nodes) == 1 else (rank - 1) / denominator,
            )
            for rank, ranked_node in enumerate(sorted_nodes, start=1)
        ]
