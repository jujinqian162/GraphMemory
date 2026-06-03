from __future__ import annotations

import random
from collections import Counter
from dataclasses import asdict, dataclass

from graph_memory.indexes.bm25 import BM25TaskRetriever
from graph_memory.indexes.dense import DenseTaskRetriever
from graph_memory.learned.features import RetrieverSeedSignalProvider, SeedSignalProvider
from graph_memory.contracts.common import TaskId, TrainPairSampleType
from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.tasks import MemoryTaskInput, MemoryTaskLabels
from graph_memory.contracts.training_pairs import TrainPairBuildSummary, TrainPairRecord
from graph_memory.types import (
    DenseConfig,
    NegativeSamplingConfig,
    RankedNode,
    Retriever,
)
from graph_memory.validation import (
    validate_graphs,
    validate_memory_task_inputs,
    validate_memory_task_labels,
    validate_negative_sampling_config,
    validate_train_pair_build_summary,
    validate_train_pairs,
    validate_task_id_alignment,
)


@dataclass(frozen=True)
class TrainPairBuildResult:
    """
    In-memory result of deterministic train pair construction.
    确定性训练 pair 构造的内存结果。

    Fields / 字段:
    - pairs: Validated training pair artifact records ready to write as JSON.
      pairs：已验证、可写入 JSON 的训练 pair artifact 记录。
    - summary: Reproducibility summary for sampling counts and effective config.
      summary：记录采样计数和实际配置的可复现性 summary。
    """

    pairs: list[TrainPairRecord]
    summary: TrainPairBuildSummary


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

    Args / 参数:
    - task_inputs: Input-visible memory tasks.
      task_inputs：retrieval 可见的 memory task 输入。
    - labels: Label artifacts that provide gold evidence nodes.
      labels：提供 gold evidence nodes 的 label artifact。
    - graphs: Graph artifacts used for graph-neighbor negatives.
      graphs：用于 graph-neighbor 负采样的 graph artifact。
    - config: Deterministic negative sampling configuration.
      config：确定性负采样配置。
    - bm25_retriever: Optional retriever override for BM25 hard negatives.
      bm25_retriever：用于 BM25 hard negative 的可选 retriever 覆盖。
    - dense_retriever: Optional retriever override for dense hard negatives.
      dense_retriever：用于 dense hard negative 的可选 retriever 覆盖。
    - dense_seed_signal_provider: Optional seed signal provider for dense hard negatives.
      dense_seed_signal_provider：用于 dense hard negative 的可选 seed signal provider。
    - dense_config: Optional dense retriever config for hard dense negatives.
      dense_config：用于 hard dense 负采样的可选 dense retriever 配置。
    """

    validate_negative_sampling_config(config)
    validate_memory_task_inputs(task_inputs)
    inputs_by_task_id = {task_input["task_id"]: task_input for task_input in task_inputs}
    labels_by_task_id = {label["task_id"]: label for label in labels}
    graphs_by_task_id = {graph["task_id"]: graph for graph in graphs}
    validate_memory_task_labels(labels, inputs_by_task_id)
    validate_graphs(graphs, inputs_by_task_id)
    validate_task_id_alignment("train pair labels", set(inputs_by_task_id), set(labels_by_task_id))
    validate_task_id_alignment("train pair graphs", set(inputs_by_task_id), set(graphs_by_task_id))

    rng = random.Random(config.random_seed)
    bm25 = bm25_retriever
    dense = dense_retriever
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
        non_gold_node_ids = [node_id for node_id in memory_node_ids if node_id not in gold_node_set]

        _append_negative_samples(
            pairs,
            seen_pair_keys,
            negative_count_by_type,
            task_id=task_id,
            node_ids=_sample_random(non_gold_node_ids, config.easy_random_per_positive * len(gold_nodes), rng),
            sample_type="easy_random",
        )

        if config.hard_bm25_per_positive > 0:
            bm25 = bm25 or BM25TaskRetriever()
            _append_negative_samples(
                pairs,
                seen_pair_keys,
                negative_count_by_type,
                task_id=task_id,
                node_ids=_hard_retriever_negatives(
                    bm25.rank(task_input),
                    gold_node_set,
                    desired_count=config.hard_bm25_per_positive * len(gold_nodes),
                    hard_pool_size=config.hard_pool_size,
                ),
                sample_type="hard_bm25",
            )

        if config.hard_dense_per_positive > 0:
            if dense_seed_signal_provider is not None:
                dense_provider = dense_seed_signal_provider
            else:
                if dense is None and dense_config is not None:
                    dense = DenseTaskRetriever(
                        model_name=dense_config.model_name,
                        batch_size=dense_config.batch_size,
                        query_prefix=dense_config.query_prefix,
                        passage_prefix=dense_config.passage_prefix,
                    )
                else:
                    dense = dense or DenseTaskRetriever()
                dense_provider = RetrieverSeedSignalProvider(dense)
            _append_negative_samples(
                pairs,
                seen_pair_keys,
                negative_count_by_type,
                task_id=task_id,
                node_ids=_hard_retriever_negatives(
                    [
                        RankedNode(node_id=signal.node_id, score=signal.score)
                        for signal in dense_provider.score_task(task_input)
                    ],
                    gold_node_set,
                    desired_count=config.hard_dense_per_positive * len(gold_nodes),
                    hard_pool_size=config.hard_pool_size,
                ),
                sample_type="hard_dense",
            )

        _append_negative_samples(
            pairs,
            seen_pair_keys,
            negative_count_by_type,
            task_id=task_id,
            node_ids=_graph_neighbor_negatives(
                task_graph,
                gold_node_set,
                non_gold_node_ids,
                desired_count=config.hard_graph_neighbor_per_positive * len(gold_nodes),
            ),
            sample_type="hard_graph_neighbor",
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
        "sampling_config": asdict(config),
    }

    validate_train_pairs(
        pairs,
        inputs_by_task_id,
        labels_by_task_id,
        graphs_by_task_id,
    )
    validate_train_pair_build_summary(summary)
    return TrainPairBuildResult(pairs=pairs, summary=summary)


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


def _sample_random(node_ids: list[str], desired_count: int, rng: random.Random) -> list[str]:
    if desired_count <= 0 or not node_ids:
        return []
    count = min(desired_count, len(node_ids))
    return rng.sample(sorted(node_ids), count)


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


def _graph_neighbor_negatives(
    graph: MemoryGraph,
    gold_node_ids: set[str],
    non_gold_node_ids: list[str],
    *,
    desired_count: int,
) -> list[str]:
    if desired_count <= 0:
        return []
    non_gold_node_id_set = set(non_gold_node_ids)
    candidates: list[str] = []
    for edge in graph["edges"]:
        source = edge["source"]
        target = edge["target"]
        if source in gold_node_ids and target in non_gold_node_id_set:
            candidates.append(target)
        if target in gold_node_ids and source in non_gold_node_id_set:
            candidates.append(source)
    return _deduplicate_preserve_order(candidates)[:desired_count]


def _deduplicate_preserve_order(node_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for node_id in node_ids:
        if node_id in seen:
            continue
        seen.add(node_id)
        unique.append(node_id)
    return unique
