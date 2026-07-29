from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import torch
import torch.nn.functional as F

from graph_memory.graphs.contracts import EvidenceGraph
from graph_memory.evaluation.contracts import MetricRow
from graph_memory.retrieval.results import RankedResult
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.graphs.views import induced_edges, model_visible_graph
from graph_memory.models.graph_retriever.batching import (
    build_evidence_dataloader,
    materialize_full_ranking_tasks,
    move_training_batch,
)
from graph_memory.models.graph_retriever.config.records import RgcnModelConfig
from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
from graph_memory.models.graph_retriever.internals.contracts import TrainingBatch
from graph_memory.models.graph_retriever.internals.neural import EvidenceScoringModel
from graph_memory.retrieval.contracts import RankedNode
from graph_memory.retrieval.execution.results import assemble_ranked_result
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.retrieval.results import RankedResultEnvelope
from graph_memory.retrieval.signals import SeedSignalProvider


def predict_dev(
    *,
    model: EvidenceScoringModel,
    ranking_requests: list[TextRankingRequest],
    labels: list[EvidenceLabel],
    graphs: list[EvidenceGraph],
    model_config: RgcnModelConfig,
    text_embedding_provider: TextEmbeddingProvider,
    seed_signal_provider: SeedSignalProvider,
    per_device_graph_batch_size: int,
    device: torch.device,
) -> tuple[list[RankedResult], float]:
    tasks = materialize_full_ranking_tasks(
        ranking_requests=ranking_requests,
        graphs=graphs,
        model_config=model_config,
        text_embedding_provider=text_embedding_provider,
        seed_signal_provider=seed_signal_provider,
        labels=labels,
    )
    batches = build_evidence_dataloader(
        tasks,
        per_device_graph_batch_size=per_device_graph_batch_size,
        shuffle=False,
        random_seed=0,
    )
    return predict_dev_from_batches(
        model=model,
        ranking_requests=ranking_requests,
        labels=labels,
        graphs=graphs,
        model_config=model_config,
        batches=batches,
        device=device,
    )


def predict_dev_from_batches(
    *,
    model: EvidenceScoringModel,
    ranking_requests: list[TextRankingRequest],
    labels: list[EvidenceLabel],
    graphs: list[EvidenceGraph],
    model_config: RgcnModelConfig,
    batches: Iterable[TrainingBatch],
    device: torch.device,
) -> tuple[list[RankedResult], float]:
    labels_by_task_id = {label.task_id: label for label in labels}
    graph_by_task_id = {graph.task_id: graph for graph in graphs}
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
            for task_id, node_id, score in zip(
                batch.sample_task_ids,
                batch.sample_node_ids,
                logits.detach().cpu().tolist(),
            ):
                logits_by_task_id[task_id].append(
                    RankedNode(node_id=node_id, score=float(score))
                )

    predictions: list[RankedResult] = []
    for request in ranking_requests:
        task_id = request.task_id
        ranked_nodes = sorted(
            logits_by_task_id[task_id],
            key=lambda ranked_node: (-ranked_node.score, ranked_node.node_id),
        )
        top_node_ids = [ranked_node.node_id for ranked_node in ranked_nodes[:10]]
        visible_graph = model_visible_graph(
            graph_by_task_id[task_id], frozenset(model_config.enabled_edge_types)
        )
        retrieved_edges = induced_edges(visible_graph, top_node_ids)
        prediction = assemble_ranked_result(
            text_request=request,
            method=model_config.method_name,
            ranked_nodes=ranked_nodes,
            top_k=10,
            latency_ms=0.0,
            retrieved_edges=retrieved_edges,
            native_trace=None,
        )
        RankedResultEnvelope(request=request, result=prediction)
        predictions.append(prediction)
        if not set(labels_by_task_id[task_id].gold_evidence_item_ids):
            raise ValueError(
                f"Dev labels must contain gold evidence nodes for task_id={task_id}."
            )
    return predictions, loss_total / sample_count if sample_count else 0.0


def best_metric(row: MetricRow) -> float:
    return (
        0.50 * row.full_support_at_5
        + 0.30 * row.recall_at_5
        + 0.20 * row.mrr
    )
