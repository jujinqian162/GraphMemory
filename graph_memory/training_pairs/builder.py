from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence

from tqdm.auto import tqdm

from graph_memory.contracts.common import TaskId, TrainPairSampleType
from graph_memory.retrieval.bulk import task_groups
from graph_memory.retrieval.contracts import RankedNode
from graph_memory.retrieval.methods.flat.bm25 import BM25TaskRetriever
from graph_memory.retrieval.methods.flat.dense import DenseConfig, DenseTaskRetriever
from graph_memory.retrieval.signals import (
    RetrieverSeedSignalProvider,
    SeedSignal,
    score_tasks,
)
from graph_memory.training_pairs.config import NegativeSamplingConfig
from graph_memory.training_pairs.contracts import (
    TrainPairBuildResult,
    TrainPairBuildSummary,
    TrainPairDataset,
    TrainPairRecord,
)
from graph_memory.training_pairs.requests import (
    CandidateNeighborEdge,
    TrainPairBuildTask,
)


def build_train_pairs(
    tasks: Sequence[TrainPairBuildTask],
    config: NegativeSamplingConfig,
    *,
    dense_config: DenseConfig | None = None,
    progress_desc: str | None = None,
) -> TrainPairBuildResult:
    """Build deterministic positive and negative training pairs."""

    task_list = list(tasks)
    text_requests = [task.text_request for task in task_list]
    labels_by_task_id = {task.label.task_id: task.label for task in task_list}
    graphs = [task.graph for task in task_list if task.graph is not None]
    if graphs and len(graphs) != len(task_list):
        raise ValueError(
            "Train-pair tasks must either all provide evidence graphs or all omit them."
        )
    if config.hard_graph_neighbor_per_positive > 0 and any(
        _candidate_neighbor_edges(task) is None for task in task_list
    ):
        raise ValueError(
            "hard_graph_neighbor_per_positive must be zero when train-pair tasks "
            "do not provide candidate neighbor edges."
        )

    dense_signals = _dense_signals_by_task(
        text_requests,
        config=config,
        dense_config=dense_config,
    )
    bm25 = BM25TaskRetriever() if config.hard_bm25_per_positive > 0 else None
    rng = random.Random(config.random_seed)
    pairs: list[TrainPairRecord] = []
    seen_pair_keys: set[tuple[TaskId, str, TrainPairSampleType]] = set()
    negative_count_by_type: Counter[str] = Counter()
    tasks_with_no_positive: list[TaskId] = []

    task_iterator = task_list
    if progress_desc is not None:
        task_iterator = tqdm(task_list, desc=progress_desc, unit="task")
    for task in task_iterator:
        request = task.text_request
        task_id = request.task_id
        gold_node_ids = list(task.label.gold_evidence_item_ids)
        if not gold_node_ids:
            tasks_with_no_positive.append(task_id)
            continue

        for node_id in gold_node_ids:
            _append_pair(
                pairs,
                seen_pair_keys,
                task_id=task_id,
                node_id=node_id,
                label=1,
                sample_type="positive",
            )

        positive_count = len(gold_node_ids)
        gold = set(gold_node_ids)
        non_gold = [
            candidate.item_id
            for candidate in request.candidates
            if candidate.item_id not in gold
        ]
        easy_count = config.easy_random_per_positive * positive_count
        easy_nodes = (
            rng.sample(sorted(non_gold), min(easy_count, len(non_gold)))
            if easy_count > 0 and non_gold
            else []
        )
        _append_negative_samples(
            pairs,
            seen_pair_keys,
            negative_count_by_type,
            task_id=task_id,
            node_ids=easy_nodes,
            sample_type="easy_random",
        )

        if bm25 is not None:
            bm25_nodes = _hard_negatives(
                bm25.rank(request),
                gold,
                desired_count=config.hard_bm25_per_positive * positive_count,
                hard_pool_size=config.hard_pool_size,
            )
            _append_negative_samples(
                pairs,
                seen_pair_keys,
                negative_count_by_type,
                task_id=task_id,
                node_ids=bm25_nodes,
                sample_type="hard_bm25",
            )

        if dense_signals:
            dense_nodes = _hard_negatives(
                [
                    RankedNode(node_id=signal.node_id, score=signal.score)
                    for signal in dense_signals[task_id]
                ],
                gold,
                desired_count=config.hard_dense_per_positive * positive_count,
                hard_pool_size=config.hard_pool_size,
            )
            _append_negative_samples(
                pairs,
                seen_pair_keys,
                negative_count_by_type,
                task_id=task_id,
                node_ids=dense_nodes,
                sample_type="hard_dense",
            )

        graph_nodes = _graph_neighbor_negatives(
            _candidate_neighbor_edges(task),
            gold_node_ids=gold,
            non_gold_node_ids=set(non_gold),
            desired_count=config.hard_graph_neighbor_per_positive * positive_count,
        )
        _append_negative_samples(
            pairs,
            seen_pair_keys,
            negative_count_by_type,
            task_id=task_id,
            node_ids=graph_nodes,
            sample_type="hard_graph_neighbor",
        )

    positive_count = sum(1 for pair in pairs if pair.label == 1)
    negative_count = sum(negative_count_by_type.values())
    task_count = len(task_list)
    summary = TrainPairBuildSummary(
        positive_count=positive_count,
        negative_count_by_type=dict(sorted(negative_count_by_type.items())),
        avg_positive_per_task=positive_count / task_count if task_count else 0.0,
        avg_negative_per_task=negative_count / task_count if task_count else 0.0,
        tasks_with_no_positive=tuple(tasks_with_no_positive),
        sampling_config=config,
    )
    dataset = TrainPairDataset(
        requests=tuple(text_requests),
        labels=tuple(labels_by_task_id.values()),
        graphs=tuple(graph for graph in graphs if graph is not None),
        pairs=tuple(pairs),
    )
    return TrainPairBuildResult(pairs=dataset.pairs, summary=summary)


def _dense_signals_by_task(
    requests,
    *,
    config: NegativeSamplingConfig,
    dense_config: DenseConfig | None,
) -> dict[str, list[SeedSignal]]:
    if config.hard_dense_per_positive <= 0:
        return {}
    if dense_config is None:
        raise ValueError("Hard dense sampling requires dense_config.")
    provider = RetrieverSeedSignalProvider(
        DenseTaskRetriever(
            config=dense_config,
            device=dense_config.device,
        )
    )
    signals_by_task_id: dict[str, list[SeedSignal]] = {}
    for request_group in task_groups(requests):
        for request, signals in zip(
            request_group,
            score_tasks(provider, request_group),
            strict=True,
        ):
            signals_by_task_id[request.task_id] = signals
    return signals_by_task_id


def _candidate_neighbor_edges(
    task: TrainPairBuildTask,
) -> tuple[CandidateNeighborEdge, ...] | None:
    if task.candidate_neighbor_edges is not None:
        return task.candidate_neighbor_edges
    if task.graph is None:
        return None
    return tuple(
        CandidateNeighborEdge(source=edge.source, target=edge.target)
        for edge in task.graph.edges
    )


def _hard_negatives(
    ranked_nodes: list[RankedNode],
    gold_node_ids: set[str],
    *,
    desired_count: int,
    hard_pool_size: int,
) -> list[str]:
    if desired_count <= 0:
        return []
    pool = [
        ranked_node.node_id
        for ranked_node in ranked_nodes
        if ranked_node.node_id not in gold_node_ids
    ]
    return _deduplicate(pool[:hard_pool_size])[:desired_count]


def _graph_neighbor_negatives(
    edges: tuple[CandidateNeighborEdge, ...] | None,
    *,
    gold_node_ids: set[str],
    non_gold_node_ids: set[str],
    desired_count: int,
) -> list[str]:
    if desired_count <= 0:
        return []
    if edges is None:
        raise ValueError(
            "Graph-neighbor negative sampling requires candidate neighbor edges."
        )
    candidates: list[str] = []
    for edge in edges:
        if edge.source in gold_node_ids and edge.target in non_gold_node_ids:
            candidates.append(edge.target)
        if edge.target in gold_node_ids and edge.source in non_gold_node_ids:
            candidates.append(edge.source)
    return _deduplicate(candidates)[:desired_count]


def _deduplicate(node_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(node_ids))


def _append_pair(
    pairs: list[TrainPairRecord],
    seen_pair_keys: set[tuple[TaskId, str, TrainPairSampleType]],
    *,
    task_id: TaskId,
    node_id: str,
    label: int,
    sample_type: TrainPairSampleType,
) -> bool:
    pair_key = (task_id, node_id, sample_type)
    if pair_key in seen_pair_keys:
        return False
    seen_pair_keys.add(pair_key)
    pairs.append(
        TrainPairRecord(
            task_id=task_id,
            node_id=node_id,
            label=1 if label == 1 else 0,
            sample_type=sample_type,
        )
    )
    return True


def _append_negative_samples(
    pairs: list[TrainPairRecord],
    seen_pair_keys: set[tuple[TaskId, str, TrainPairSampleType]],
    negative_count_by_type: Counter[str],
    *,
    task_id: TaskId,
    node_ids: list[str],
    sample_type: TrainPairSampleType,
) -> None:
    for node_id in node_ids:
        if _append_pair(
            pairs,
            seen_pair_keys,
            task_id=task_id,
            node_id=node_id,
            label=0,
            sample_type=sample_type,
        ):
            negative_count_by_type[sample_type] += 1
