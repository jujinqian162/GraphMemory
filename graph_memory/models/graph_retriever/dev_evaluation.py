from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.metrics import MetricRow
from graph_memory.contracts.ranking import RankedResult
from graph_memory.contracts.tasks import MemoryTaskInput, MemoryTaskLabels
from graph_memory.graphs.views import induced_retrieved_subgraph, model_visible_graph
from graph_memory.models.graph_retriever.batching import build_full_ranking_batches, move_training_batch
from graph_memory.models.graph_retriever.config.records import TrainableModelConfig
from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
from graph_memory.models.graph_retriever.internals.neural import EvidenceScoringModel
from graph_memory.models.graph_retriever.internals.contracts import TrainingBatch
from graph_memory.retrieval.contracts import RankedNode
from graph_memory.retrieval.signals import SeedSignalProvider


def predict_dev(
    *,
    model: EvidenceScoringModel,
    task_inputs: list[MemoryTaskInput],
    labels: list[MemoryTaskLabels],
    graphs: list[MemoryGraph],
    model_config: TrainableModelConfig,
    text_embedding_provider: TextEmbeddingProvider,
    seed_signal_provider: SeedSignalProvider,
    batch_size: int,
    device: torch.device,
) -> tuple[list[RankedResult], float]:
    batches = build_full_ranking_batches(
        task_inputs=task_inputs,
        graphs=graphs,
        model_config=model_config,
        text_embedding_provider=text_embedding_provider,
        seed_signal_provider=seed_signal_provider,
        batch_size=batch_size,
        labels=labels,
    )
    return predict_dev_from_batches(
        model=model,
        task_inputs=task_inputs,
        labels=labels,
        graphs=graphs,
        model_config=model_config,
        batches=batches,
        device=device,
    )


def predict_dev_from_batches(
    *,
    model: EvidenceScoringModel,
    task_inputs: list[MemoryTaskInput],
    labels: list[MemoryTaskLabels],
    graphs: list[MemoryGraph],
    model_config: TrainableModelConfig,
    batches: Sequence[TrainingBatch],
    device: torch.device,
) -> tuple[list[RankedResult], float]:
    labels_by_task_id = {label["task_id"]: label for label in labels}
    graph_by_task_id = {graph["task_id"]: graph for graph in graphs}
    logits_by_task_id: dict[str, list[RankedNode]] = defaultdict(list)
    loss_total = 0.0
    sample_count = 0

    model.eval()
    with torch.no_grad():
        for batch in batches:
            moved_batch = move_training_batch(batch, device)
            logits = model(moved_batch)
            loss = F.binary_cross_entropy_with_logits(logits, moved_batch.labels)
            loss_total += float(loss.detach().cpu()) * int(moved_batch.labels.shape[0])
            sample_count += int(moved_batch.labels.shape[0])
            for task_id, node_id, score in zip(batch.sample_task_ids, batch.sample_node_ids, logits.detach().cpu().tolist()):
                logits_by_task_id[task_id].append(RankedNode(node_id=node_id, score=float(score)))

    predictions: list[RankedResult] = []
    for task_input in task_inputs:
        task_id = task_input["task_id"]
        ranked_nodes = sorted(logits_by_task_id[task_id], key=lambda ranked_node: (-ranked_node.score, ranked_node.node_id))
        top_node_ids = [ranked_node.node_id for ranked_node in ranked_nodes[:10]]
        visible_graph = model_visible_graph(graph_by_task_id[task_id], frozenset(model_config.enabled_edge_types))
        subgraph = induced_retrieved_subgraph(visible_graph, top_node_ids)
        predictions.append(
            {
                "task_id": task_id,
                "method": model_config.method_name,
                "ranked_nodes": [{"node_id": node.node_id, "score": node.score} for node in ranked_nodes],
                "retrieved_subgraph": subgraph,
                "latency_ms": 0.0,
                "input_tokens": 0,
            }
        )
        if not set(labels_by_task_id[task_id]["gold_evidence_nodes"]):
            raise ValueError(f"Dev labels must contain gold evidence nodes for task_id={task_id}.")
    return predictions, loss_total / sample_count if sample_count else 0.0


def best_metric(row: MetricRow) -> float:
    return 0.50 * _metric_float(row, "Full Support@5") + 0.30 * _metric_float(row, "Recall@5") + 0.20 * _metric_float(row, "MRR")


def _metric_float(row: MetricRow, key: str) -> float:
    value = row[key]
    if not isinstance(value, (int, float)):
        raise ValueError(f"Metric column must be numeric for best checkpoint selection: {key}")
    return float(value)
