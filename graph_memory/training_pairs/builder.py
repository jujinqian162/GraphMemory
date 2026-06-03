from __future__ import annotations

import random
from collections import Counter
from dataclasses import asdict, dataclass

from graph_memory.contracts.common import TaskId, TrainPairSampleType
from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.tasks import MemoryTaskInput, MemoryTaskLabels
from graph_memory.contracts.training_pairs import TrainPairBuildSummary, TrainPairRecord
from graph_memory.retrieval.contracts import Retriever
from graph_memory.retrieval.methods.flat.bm25 import BM25TaskRetriever
from graph_memory.retrieval.methods.flat.dense import DenseConfig, DenseTaskRetriever
from graph_memory.retrieval.signals import RetrieverSeedSignalProvider, SeedSignalProvider
from graph_memory.training_pairs.config import NegativeSamplingConfig
from graph_memory.training_pairs.samplers import (
    BM25HardNegativeSampler,
    DenseHardNegativeSampler,
    EasyRandomNegativeSampler,
    GraphNeighborNegativeSampler,
    NegativeSampler,
    PairSamplingContext,
)
from graph_memory.validation import (
    validate_graphs,
    validate_memory_task_inputs,
    validate_memory_task_labels,
    validate_negative_sampling_config,
    validate_task_id_alignment,
    validate_train_pair_build_summary,
    validate_train_pairs,
)


@dataclass(frozen=True)
class TrainPairBuildResult:
    """
    In-memory result of deterministic train pair construction.
    确定性训练 pair 构造的内存结果。
    """

    pairs: list[TrainPairRecord]
    summary: TrainPairBuildSummary


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
        task_inputs: list[MemoryTaskInput],
        labels: list[MemoryTaskLabels],
        graphs: list[MemoryGraph],
    ) -> TrainPairBuildResult:
        validate_negative_sampling_config(self.config)
        validate_memory_task_inputs(task_inputs)
        inputs_by_task_id = {task_input["task_id"]: task_input for task_input in task_inputs}
        labels_by_task_id = {label["task_id"]: label for label in labels}
        graphs_by_task_id = {graph["task_id"]: graph for graph in graphs}
        validate_memory_task_labels(labels, inputs_by_task_id)
        validate_graphs(graphs, inputs_by_task_id)
        validate_task_id_alignment("train pair labels", set(inputs_by_task_id), set(labels_by_task_id))
        validate_task_id_alignment("train pair graphs", set(inputs_by_task_id), set(graphs_by_task_id))

        rng = random.Random(self.config.random_seed)
        pairs: list[TrainPairRecord] = []
        seen_pair_keys: set[tuple[TaskId, str, TrainPairSampleType]] = set()
        negative_count_by_type: Counter[str] = Counter()
        tasks_with_no_positive: list[TaskId] = []

        for task_input in task_inputs:
            task_id = task_input["task_id"]
            task_labels = labels_by_task_id[task_id]
            task_graph = graphs_by_task_id[task_id]
            memory_node_ids = [memory_item["id"] for memory_item in task_input["memory_items"]]
            gold_nodes = list(task_labels["gold_evidence_nodes"])
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
                task_input=task_input,
                graph=task_graph,
                gold_node_ids=gold_node_set,
                non_gold_node_ids=[node_id for node_id in memory_node_ids if node_id not in gold_node_set],
                rng=rng,
            )

            for sampler in self.samplers:
                desired_count = _desired_count(self.config, sampler.sample_type, positive_count=len(gold_nodes))
                _append_negative_samples(
                    pairs,
                    seen_pair_keys,
                    negative_count_by_type,
                    task_id=task_id,
                    node_ids=sampler.sample(context, desired_count),
                    sample_type=sampler.sample_type,
                )

        positive_count = sum(1 for pair in pairs if pair["label"] == 1)
        negative_count = sum(negative_count_by_type.values())
        num_tasks = len(task_inputs)
        summary: TrainPairBuildSummary = {
            "positive_count": positive_count,
            "negative_count_by_type": dict(sorted(negative_count_by_type.items())),
            "avg_positive_per_task": positive_count / num_tasks if num_tasks else 0.0,
            "avg_negative_per_task": negative_count / num_tasks if num_tasks else 0.0,
            "tasks_with_no_positive": tasks_with_no_positive,
            "sampling_config": asdict(self.config),
        }

        validate_train_pairs(
            pairs,
            inputs_by_task_id,
            labels_by_task_id,
            graphs_by_task_id,
        )
        validate_train_pair_build_summary(summary)
        return TrainPairBuildResult(pairs=pairs, summary=summary)


def build_train_pairs(
    task_inputs: list[MemoryTaskInput],
    labels: list[MemoryTaskLabels],
    graphs: list[MemoryGraph],
    config: NegativeSamplingConfig,
    *,
    bm25_retriever: Retriever | None = None,
    dense_retriever: Retriever | None = None,
    dense_seed_signal_provider: SeedSignalProvider | None = None,
    dense_config: DenseConfig | None = None,
) -> TrainPairBuildResult:
    """
    Build validated train pair records from already-loaded artifacts.
    从已读取的 artifact 构造并验证训练 pair 记录。
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
    return builder.build(task_inputs, labels, graphs)


def _build_default_samplers(
    config: NegativeSamplingConfig,
    *,
    bm25_retriever: Retriever | None,
    dense_retriever: Retriever | None,
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
            if retriever is None and dense_config is not None:
                retriever = DenseTaskRetriever(config=dense_config)
            else:
                retriever = retriever or DenseTaskRetriever()
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
        {
            "task_id": task_id,
            "node_id": node_id,
            "label": 1 if label == 1 else 0,
            "sample_type": sample_type,
        }
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
