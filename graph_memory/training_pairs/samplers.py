from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from graph_memory.contracts.common import TrainPairSampleType
from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.retrieval.bulk import task_groups
from graph_memory.retrieval.contracts import RankedNode, SeedRanker
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.retrieval.signals import SeedSignal, SeedSignalProvider, score_tasks


@dataclass(frozen=True)
class PairSamplingContext:
    """
    Per-task state shared by negative samplers.
    单个 task 的负采样共享状态。
    """

    text_request: TextRankingRequest
    graph: MemoryGraph
    gold_node_ids: set[str]
    non_gold_node_ids: list[str]
    rng: random.Random


class NegativeSampler(Protocol):
    @property
    def sample_type(self) -> TrainPairSampleType:
        ...

    def sample(self, context: PairSamplingContext, desired_count: int) -> list[str]:
        ...


@dataclass(frozen=True)
class EasyRandomNegativeSampler:
    sample_type: TrainPairSampleType = "easy_random"

    def sample(self, context: PairSamplingContext, desired_count: int) -> list[str]:
        if desired_count <= 0 or not context.non_gold_node_ids:
            return []
        count = min(desired_count, len(context.non_gold_node_ids))
        return context.rng.sample(sorted(context.non_gold_node_ids), count)


@dataclass(frozen=True)
class BM25HardNegativeSampler:
    retriever: SeedRanker
    hard_pool_size: int
    sample_type: TrainPairSampleType = "hard_bm25"

    def sample(self, context: PairSamplingContext, desired_count: int) -> list[str]:
        return _hard_retriever_negatives(
            self.retriever.rank(context.text_request),
            context.gold_node_ids,
            desired_count=desired_count,
            hard_pool_size=self.hard_pool_size,
        )


@dataclass(frozen=True)
class DenseHardNegativeSampler:
    seed_signal_provider: SeedSignalProvider
    hard_pool_size: int
    precomputed_signals_by_task_id: Mapping[str, list[SeedSignal]] | None = None
    sample_type: TrainPairSampleType = "hard_dense"

    def sample(self, context: PairSamplingContext, desired_count: int) -> list[str]:
        signals = (
            self.precomputed_signals_by_task_id.get(context.text_request.task_id)
            if self.precomputed_signals_by_task_id is not None
            else None
        )
        if signals is None:
            signals = self.seed_signal_provider.score_task(context.text_request)
        return _hard_retriever_negatives(
            [
                RankedNode(node_id=signal.node_id, score=signal.score)
                for signal in signals
            ],
            context.gold_node_ids,
            desired_count=desired_count,
            hard_pool_size=self.hard_pool_size,
        )

    def precompute(
        self,
        text_requests: Sequence[TextRankingRequest],
    ) -> DenseHardNegativeSampler:
        signals_by_task_id: dict[str, list[SeedSignal]] = {}
        for request_group in task_groups(text_requests):
            for request, signals in zip(
                request_group,
                score_tasks(self.seed_signal_provider, request_group),
                strict=True,
            ):
                signals_by_task_id[request.task_id] = signals
        return replace(self, precomputed_signals_by_task_id=signals_by_task_id)


@dataclass(frozen=True)
class GraphNeighborNegativeSampler:
    sample_type: TrainPairSampleType = "hard_graph_neighbor"

    def sample(self, context: PairSamplingContext, desired_count: int) -> list[str]:
        if desired_count <= 0:
            return []
        non_gold_node_id_set = set(context.non_gold_node_ids)
        candidates: list[str] = []
        for edge in context.graph["edges"]:
            source = edge["source"]
            target = edge["target"]
            if source in context.gold_node_ids and target in non_gold_node_id_set:
                candidates.append(target)
            if target in context.gold_node_ids and source in non_gold_node_id_set:
                candidates.append(source)
        return _deduplicate_preserve_order(candidates)[:desired_count]


def _hard_retriever_negatives(
    ranked_nodes: list[RankedNode],
    gold_node_ids: set[str],
    *,
    desired_count: int,
    hard_pool_size: int,
) -> list[str]:
    if desired_count <= 0:
        return []
    pool = [ranked_node.node_id for ranked_node in ranked_nodes if ranked_node.node_id not in gold_node_ids]
    return _deduplicate_preserve_order(pool[:hard_pool_size])[:desired_count]


def _deduplicate_preserve_order(node_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for node_id in node_ids:
        if node_id in seen:
            continue
        seen.add(node_id)
        unique.append(node_id)
    return unique
