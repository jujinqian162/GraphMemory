from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from graph_memory.contracts.common import NodeId
from graph_memory.retrieval.contracts import RankedNode, SeedRanker
from graph_memory.retrieval.requests import TextRankingRequest


@dataclass(frozen=True)
class SeedSignal:
    """
    Frozen seed retrieval signal for one candidate node.
    一个候选节点的冻结初始检索信号。
    """

    node_id: NodeId
    score: float
    rank: int
    rank_percentile: float


@runtime_checkable
class SeedSignalProvider(Protocol):
    """
    Replaceable provider for frozen seed retrieval signals.
    可替换的冻结初始检索信号提供器。
    """

    def score_task(self, request: TextRankingRequest) -> list[SeedSignal]:
        ...


@runtime_checkable
class BulkSeedSignalProvider(Protocol):
    def score_tasks(self, requests: Sequence[TextRankingRequest]) -> list[list[SeedSignal]]:
        ...


def score_tasks(
    provider: SeedSignalProvider,
    requests: Sequence[TextRankingRequest],
) -> list[list[SeedSignal]]:
    request_list = list(requests)
    if isinstance(provider, BulkSeedSignalProvider):
        results = provider.score_tasks(request_list)
        if len(results) != len(request_list):
            raise ValueError(
                "Bulk seed signal provider returned an invalid result count: "
                f"expected={len(request_list)} observed={len(results)}."
            )
        return results
    return [provider.score_task(request) for request in request_list]


def seed_signals_from_ranked_nodes(
    request: TextRankingRequest,
    ranked_nodes: list[RankedNode],
) -> list[SeedSignal]:
    expected_node_ids = {candidate.item_id for candidate in request.candidates}
    observed_node_ids = {ranked_node.node_id for ranked_node in ranked_nodes}
    if observed_node_ids != expected_node_ids:
        missing = sorted(expected_node_ids - observed_node_ids)
        extra = sorted(observed_node_ids - expected_node_ids)
        raise ValueError(f"Seed retriever must return every candidate node exactly once; missing={missing} extra={extra}.")

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


@dataclass(frozen=True)
class RetrieverSeedSignalProvider:
    """
    Seed signal provider backed by an existing flat retriever.
    基于现有 flat retriever 的 seed signal provider。
    """

    retriever: SeedRanker

    def score_task(self, request: TextRankingRequest) -> list[SeedSignal]:
        return seed_signals_from_ranked_nodes(request, self.retriever.rank(request))

    def score_tasks(self, requests: Sequence[TextRankingRequest]) -> list[list[SeedSignal]]:
        from graph_memory.retrieval.bulk import rank_tasks

        return [
            seed_signals_from_ranked_nodes(request, ranked_nodes)
            for request, ranked_nodes in zip(
                requests,
                rank_tasks(self.retriever, requests),
                strict=True,
            )
        ]
