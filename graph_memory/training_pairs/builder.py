from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from tqdm.auto import tqdm

from graph_memory.contracts.common import TaskId, TrainPairSampleType
from graph_memory.training_pairs.contracts import (
    TrainPairBuildResult,
    TrainPairBuildSummary,
    TrainPairDataset,
    TrainPairRecord,
)
from graph_memory.retrieval.contracts import SeedRanker
from graph_memory.retrieval.methods.flat.bm25 import BM25TaskRetriever
from graph_memory.retrieval.methods.flat.dense import DenseConfig, DenseTaskRetriever
from graph_memory.retrieval.signals import RetrieverSeedSignalProvider, SeedSignalProvider
from graph_memory.training_pairs.config import (
    NegativeSamplingConfig,
    ProvenanceNegativeSamplingConfig,
)
from graph_memory.training_pairs.requests import (
    ProvenanceTrainPairBuildTask,
    TrainPairBuildTask,
)
from graph_memory.training_pairs.samplers import (
    BM25HardNegativeSampler,
    DenseHardNegativeSampler,
    EasyRandomNegativeSampler,
    GraphNeighborNegativeSampler,
    NegativeSampler,
    PairSamplingContext,
    ProvenancePredecessorNegativeSampler,
    ProvenanceSuccessorNegativeSampler,
)
@dataclass(frozen=True)
class TrainPairBuilder:
    """
    Coordinates deterministic train pair construction.
    协调确定性训练 pair 构造。
    """

    config: NegativeSamplingConfig
    samplers: tuple[NegativeSampler, ...]

    def build(
        self,
        tasks: Sequence[TrainPairBuildTask],
        *,
        progress_desc: str | None = None,
    ) -> TrainPairBuildResult:
        task_list = list(tasks)
        text_requests = [task.text_request for task in task_list]
        labels_by_task_id = {task.label.task_id: task.label for task in task_list}
        graphs = [task.graph for task in task_list if task.graph is not None]
        if graphs and len(graphs) != len(task_list):
            raise ValueError(
                "Train-pair tasks must either all provide evidence graphs or all omit them."
            )
        graphs_by_task_id = {graph.task_id: graph for graph in graphs}
        if (
            not graphs_by_task_id
            and self.config.hard_graph_neighbor_per_positive > 0
        ):
            raise ValueError(
                "hard_graph_neighbor_per_positive must be zero when train-pair tasks "
                "do not provide evidence graphs."
            )

        rng = random.Random(self.config.random_seed)
        pairs: list[TrainPairRecord] = []
        seen_pair_keys: set[tuple[TaskId, str, TrainPairSampleType]] = set()
        negative_count_by_type: Counter[str] = Counter()
        tasks_with_no_positive: list[TaskId] = []
        prepared_samplers = tuple(
            sampler.precompute(text_requests)
            if isinstance(sampler, DenseHardNegativeSampler)
            else sampler
            for sampler in self.samplers
        )

        task_iterator = task_list
        if progress_desc is not None:
            task_iterator = tqdm(task_list, desc=progress_desc, unit="task")
        for task in task_iterator:
            text_request = task.text_request
            task_id = text_request.task_id
            memory_node_ids = [candidate.item_id for candidate in text_request.candidates]
            gold_nodes = list(task.label.gold_evidence_item_ids)
            if not gold_nodes:
                tasks_with_no_positive.append(task_id)
                continue

            for node_id in gold_nodes:
                _append_pair(
                    pairs,
                    seen_pair_keys,
                    task_id=task_id,
                    node_id=node_id,
                    label=1,
                    sample_type="positive",
                )

            gold_node_set = set(gold_nodes)
            context = PairSamplingContext(
                text_request=text_request,
                graph=task.graph,
                gold_node_ids=gold_node_set,
                non_gold_node_ids=[node_id for node_id in memory_node_ids if node_id not in gold_node_set],
                rng=rng,
            )

            for sampler in prepared_samplers:
                desired_count = _desired_count(self.config, sampler.sample_type, positive_count=len(gold_nodes))
                _append_negative_samples(
                    pairs,
                    seen_pair_keys,
                    negative_count_by_type,
                    task_id=task_id,
                    node_ids=sampler.sample(context, desired_count),
                    sample_type=sampler.sample_type,
                )

        positive_count = sum(1 for pair in pairs if pair.label == 1)
        negative_count = sum(negative_count_by_type.values())
        num_tasks = len(task_list)
        summary = TrainPairBuildSummary(
            positive_count=positive_count,
            negative_count_by_type=dict(sorted(negative_count_by_type.items())),
            avg_positive_per_task=(
                positive_count / num_tasks if num_tasks else 0.0
            ),
            avg_negative_per_task=(
                negative_count / num_tasks if num_tasks else 0.0
            ),
            tasks_with_no_positive=tuple(tasks_with_no_positive),
            sampling_config=self.config,
        )
        dataset = TrainPairDataset(
            requests=tuple(text_requests),
            labels=tuple(labels_by_task_id.values()),
            graphs=tuple(graphs_by_task_id.values()),
            pairs=tuple(pairs),
        )
        return TrainPairBuildResult(pairs=dataset.pairs, summary=summary)


def build_train_pairs(
    tasks: Sequence[TrainPairBuildTask],
    config: NegativeSamplingConfig,
    *,
    bm25_retriever: SeedRanker | None = None,
    dense_retriever: SeedRanker | None = None,
    dense_seed_signal_provider: SeedSignalProvider | None = None,
    dense_config: DenseConfig | None = None,
    progress_desc: str | None = None,
) -> TrainPairBuildResult:
    """
    Build validated train pair records from already-projected domain tasks.
    从已投影的 domain task 构造并验证训练 pair 记录。
    """

    builder = TrainPairBuilder(
        config=config,
        samplers=_build_default_samplers(
            config,
            bm25_retriever=bm25_retriever,
            dense_retriever=dense_retriever,
            dense_seed_signal_provider=dense_seed_signal_provider,
            dense_config=dense_config,
        ),
    )
    return builder.build(tasks, progress_desc=progress_desc)


PROVENANCE_NEGATIVE_PRECEDENCE: tuple[TrainPairSampleType, ...] = (
    "hard_provenance_successor",
    "hard_provenance_predecessor",
    "hard_dense",
    "hard_bm25",
    "easy_random",
)


def build_provenance_train_pairs(
    tasks: Sequence[ProvenanceTrainPairBuildTask],
    config: ProvenanceNegativeSamplingConfig,
    *,
    bm25_retriever: SeedRanker | None = None,
    dense_retriever: SeedRanker | None = None,
    dense_seed_signal_provider: SeedSignalProvider | None = None,
    dense_config: DenseConfig | None = None,
    progress_desc: str | None = None,
) -> TrainPairBuildResult:
    task_list = list(tasks)
    text_requests = [task.text_request for task in task_list]
    labels_by_task_id = {task.label.task_id: task.label for task in task_list}

    semantic_samplers = tuple(
        sampler
        for sampler in _build_default_samplers(
            config,
            bm25_retriever=bm25_retriever,
            dense_retriever=dense_retriever,
            dense_seed_signal_provider=dense_seed_signal_provider,
            dense_config=dense_config,
        )
        if sampler.sample_type != "hard_graph_neighbor"
    )
    prepared_samplers = tuple(
        sampler.precompute(text_requests)
        if isinstance(sampler, DenseHardNegativeSampler)
        else sampler
        for sampler in semantic_samplers
    )
    successor_sampler = ProvenanceSuccessorNegativeSampler()
    predecessor_sampler = ProvenancePredecessorNegativeSampler()
    rng = random.Random(config.random_seed)
    pairs: list[TrainPairRecord] = []
    negative_count_by_type: Counter[str] = Counter()
    requested_by_type: Counter[str] = Counter()
    raw_count_by_type: Counter[str] = Counter()
    overlap_count_by_type: Counter[str] = Counter()
    source_overlap_by_task: dict[str, dict[str, list[str]]] = {}
    tasks_with_no_positive: list[TaskId] = []

    task_iterator = task_list
    if progress_desc is not None:
        task_iterator = tqdm(task_list, desc=progress_desc, unit="task")
    for task in task_iterator:
        request = task.text_request
        task_id = request.task_id
        gold_nodes = list(task.label.gold_evidence_item_ids)
        if not gold_nodes:
            tasks_with_no_positive.append(task_id)
            continue
        for node_id in gold_nodes:
            pairs.append(
                TrainPairRecord(
                    task_id=task_id,
                    node_id=node_id,
                    label=1,
                    sample_type="positive",
                )
            )
        gold_node_set = set(gold_nodes)
        context = PairSamplingContext(
            text_request=request,
            graph=None,
            gold_node_ids=gold_node_set,
            non_gold_node_ids=[
                candidate.item_id
                for candidate in request.candidates
                if candidate.item_id not in gold_node_set
            ],
            rng=rng,
        )
        desired_by_type: dict[TrainPairSampleType, int] = {
            "hard_provenance_successor": (
                config.hard_provenance_successor_per_positive * len(gold_nodes)
            ),
            "hard_provenance_predecessor": (
                config.hard_provenance_predecessor_per_positive * len(gold_nodes)
            ),
            "hard_dense": config.hard_dense_per_positive * len(gold_nodes),
            "hard_bm25": config.hard_bm25_per_positive * len(gold_nodes),
            "easy_random": config.easy_random_per_positive * len(gold_nodes),
        }
        selected_by_type: dict[TrainPairSampleType, list[str]] = {
            "hard_provenance_successor": successor_sampler.sample(
                task.graph,
                task.label,
                desired_by_type["hard_provenance_successor"],
            ),
            "hard_provenance_predecessor": predecessor_sampler.sample(
                task.graph,
                task.label,
                desired_by_type["hard_provenance_predecessor"],
            ),
        }
        for sampler in prepared_samplers:
            selected_by_type[sampler.sample_type] = sampler.sample(
                context, desired_by_type[sampler.sample_type]
            )
        sources_by_node: defaultdict[str, set[TrainPairSampleType]] = defaultdict(set)
        for sample_type, node_ids in selected_by_type.items():
            requested_by_type[sample_type] += desired_by_type[sample_type]
            raw_count_by_type[sample_type] += len(node_ids)
            for node_id in node_ids:
                if node_id not in gold_node_set:
                    sources_by_node[node_id].add(sample_type)
        task_overlaps: dict[str, list[str]] = {}
        for node_id, sources in sorted(sources_by_node.items()):
            ordered_sources: list[TrainPairSampleType] = [
                sample_type
                for sample_type in PROVENANCE_NEGATIVE_PRECEDENCE
                if sample_type in sources
            ]
            if len(ordered_sources) > 1:
                task_overlaps[node_id] = list(ordered_sources)
                overlap_count_by_type.update(ordered_sources)
            winner: TrainPairSampleType = ordered_sources[0]
            pairs.append(
                TrainPairRecord(
                    task_id=task_id,
                    node_id=node_id,
                    label=0,
                    sample_type=winner,
                )
            )
            negative_count_by_type[winner] += 1
        if task_overlaps:
            source_overlap_by_task[task_id] = task_overlaps

    positive_count = sum(pair.label for pair in pairs)
    negative_count = sum(negative_count_by_type.values())
    num_tasks = len(task_list)
    shortfall_by_type = {
        sample_type: max(0, requested_by_type[sample_type] - negative_count_by_type[sample_type])
        for sample_type in PROVENANCE_NEGATIVE_PRECEDENCE
    }
    summary = TrainPairBuildSummary(
        positive_count=positive_count,
        negative_count_by_type=dict(sorted(negative_count_by_type.items())),
        avg_positive_per_task=(positive_count / num_tasks if num_tasks else 0.0),
        avg_negative_per_task=(negative_count / num_tasks if num_tasks else 0.0),
        tasks_with_no_positive=tuple(tasks_with_no_positive),
        sampling_config=config,
        requested_negative_count_by_type=dict(sorted(requested_by_type.items())),
        shortfall_by_type=dict(sorted(shortfall_by_type.items())),
        overlap_count_by_type=dict(sorted(overlap_count_by_type.items())),
        source_overlap_by_task={
            task_id: {
                node_id: tuple(sources)
                for node_id, sources in overlaps.items()
            }
            for task_id, overlaps in source_overlap_by_task.items()
        },
    )
    dataset = TrainPairDataset(
        requests=tuple(text_requests),
        labels=tuple(labels_by_task_id.values()),
        pairs=tuple(pairs),
    )
    return TrainPairBuildResult(pairs=dataset.pairs, summary=summary)


def _build_default_samplers(
    config: NegativeSamplingConfig,
    *,
    bm25_retriever: SeedRanker | None,
    dense_retriever: SeedRanker | None,
    dense_seed_signal_provider: SeedSignalProvider | None,
    dense_config: DenseConfig | None,
) -> tuple[NegativeSampler, ...]:
    samplers: list[NegativeSampler] = [EasyRandomNegativeSampler()]
    if config.hard_bm25_per_positive > 0:
        samplers.append(BM25HardNegativeSampler(bm25_retriever or BM25TaskRetriever(), config.hard_pool_size))
    if config.hard_dense_per_positive > 0:
        if dense_seed_signal_provider is not None:
            dense_provider = dense_seed_signal_provider
        else:
            retriever = dense_retriever
            if retriever is None:
                if dense_config is None:
                    raise ValueError(
                        "Hard dense sampling requires dense_config or an injected retriever."
                    )
                retriever = DenseTaskRetriever(
                    config=dense_config,
                    device=dense_config.device,
                )
            dense_provider = RetrieverSeedSignalProvider(retriever)
        samplers.append(DenseHardNegativeSampler(dense_provider, config.hard_pool_size))
    samplers.append(GraphNeighborNegativeSampler())
    return tuple(samplers)


def _desired_count(config: NegativeSamplingConfig, sample_type: TrainPairSampleType, *, positive_count: int) -> int:
    per_positive_by_type = {
        "easy_random": config.easy_random_per_positive,
        "hard_bm25": config.hard_bm25_per_positive,
        "hard_dense": config.hard_dense_per_positive,
        "hard_graph_neighbor": config.hard_graph_neighbor_per_positive,
    }
    return per_positive_by_type.get(sample_type, 0) * positive_count


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
        added = _append_pair(
            pairs,
            seen_pair_keys,
            task_id=task_id,
            node_id=node_id,
            label=0,
            sample_type=sample_type,
        )
        if added:
            negative_count_by_type[sample_type] += 1
