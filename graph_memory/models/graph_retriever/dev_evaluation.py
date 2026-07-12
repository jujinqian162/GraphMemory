from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.metrics import MetricRow
from graph_memory.contracts.ranking import RankedResult
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.graphs.views import induced_retrieved_subgraph, model_visible_graph
from graph_memory.models.graph_retriever.batching import (
    build_full_ranking_batches,
    move_training_batch,
)
from graph_memory.models.graph_retriever.config.records import RgcnModelConfig
from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
from graph_memory.models.graph_retriever.internals.contracts import TrainingBatch
from graph_memory.models.graph_retriever.internals.neural import EvidenceScoringModel
from graph_memory.models.graph_retriever.decoder import (
    complete_beam_ranking,
    run_beam_search,
)
from graph_memory.retrieval.contracts import RankedNode
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.retrieval.signals import SeedSignalProvider


def predict_dev(
    *,
    model: EvidenceScoringModel,
    ranking_requests: list[TextRankingRequest],
    labels: list[EvidenceLabel],
    graphs: list[MemoryGraph],
    model_config: RgcnModelConfig,
    text_embedding_provider: TextEmbeddingProvider,
    seed_signal_provider: SeedSignalProvider,
    batch_size: int,
    device: torch.device,
) -> tuple[list[RankedResult], float]:
    batches = build_full_ranking_batches(
        ranking_requests=ranking_requests,
        graphs=graphs,
        model_config=model_config,
        text_embedding_provider=text_embedding_provider,
        seed_signal_provider=seed_signal_provider,
        batch_size=batch_size,
        labels=labels,
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
    graphs: list[MemoryGraph],
    model_config: RgcnModelConfig,
    batches: Sequence[TrainingBatch],
    device: torch.device,
) -> tuple[list[RankedResult], float]:
    labels_by_task_id = {label.task_id: label for label in labels}
    graph_by_task_id = {graph["task_id"]: graph for graph in graphs}
    logits_by_task_id: dict[str, list[RankedNode]] = defaultdict(list)
    metadata_by_task_id: dict[str, dict[str, object]] = {}
    loss_total = 0.0
    sample_count = 0

    model.eval()
    with torch.no_grad():
        for batch in batches:
            moved_batch = move_training_batch(batch, device)
            encoded = model.encode_graph(moved_batch)
            logits = model.base_score(
                node_states=encoded.node_states[moved_batch.sample_node_indices],
                query_states=encoded.node_states[moved_batch.sample_query_indices],
                sample_node_features=moved_batch.sample_node_features,
            )
            loss = F.binary_cross_entropy_with_logits(logits, moved_batch.labels)
            loss_total += float(loss.detach().cpu()) * int(moved_batch.labels.shape[0])
            sample_count += int(moved_batch.labels.shape[0])
            for task_index, task_id in enumerate(encoded.task_ids):
                candidate_ids = tuple(encoded.candidate_node_ids_by_task[task_index])
                beam_result = run_beam_search(
                    candidate_node_ids=candidate_ids,
                    score_actions=lambda selected, index=task_index: _score_actions(
                        model, encoded, index, selected
                    ),
                    config=model_config.beam_search_config,
                )
                ranking = complete_beam_ranking(
                    candidate_node_ids=candidate_ids,
                    base_logits=encoded.base_logits_for_task(task_index),
                    winner=beam_result.winner,
                )
                logits_by_task_id[task_id] = [
                    RankedNode(node_id=node_id, score=float(len(ranking) - rank_index))
                    for rank_index, node_id in enumerate(ranking)
                ]
                gold = set(labels_by_task_id[task_id].gold_evidence_item_ids)
                any_beam_coverage = any(
                    gold.issubset(
                        {candidate_ids[index] for index in hypothesis.selected_indices}
                    )
                    for hypothesis in beam_result.final_beam
                )
                final_coverage = gold.issubset(
                    {
                        candidate_ids[index]
                        for index in beam_result.winner.selected_indices
                    }
                )
                metadata_by_task_id[task_id] = {
                    "selected_sequence": [
                        candidate_ids[index]
                        for index in beam_result.winner.selected_indices
                    ],
                    "selected_count": len(beam_result.winner.selected_indices),
                    "stopped": beam_result.winner.stopped,
                    "stop_score": beam_result.winner.stop_score,
                    "beam_score": beam_result.winner.normalized_score,
                    "beam_size": model_config.beam_search_config.inference_beam_size,
                    "max_steps": model_config.beam_search_config.max_steps,
                    "any_beam_gold_coverage": any_beam_coverage,
                    "final_beam_gold_coverage": final_coverage,
                    "premature_stop": beam_result.winner.stopped and not final_coverage,
                    "required_evidence_count": len(gold),
                }

    predictions: list[RankedResult] = []
    for request in ranking_requests:
        task_id = request.task_id
        ranked_nodes = logits_by_task_id[task_id]
        top_node_ids = [ranked_node.node_id for ranked_node in ranked_nodes[:10]]
        visible_graph = model_visible_graph(
            graph_by_task_id[task_id], frozenset(model_config.enabled_edge_types)
        )
        subgraph = induced_retrieved_subgraph(visible_graph, top_node_ids)
        predictions.append(
            {
                "task_id": task_id,
                "method": model_config.method_name,
                "ranked_nodes": [
                    {"node_id": node.node_id, "score": node.score}
                    for node in ranked_nodes
                ],
                "retrieved_subgraph": subgraph,
                "latency_ms": 0.0,
                "input_tokens": 0,
                "metadata": metadata_by_task_id[task_id],
            }
        )
        if not set(labels_by_task_id[task_id].gold_evidence_item_ids):
            raise ValueError(
                f"Dev labels must contain gold evidence nodes for task_id={task_id}."
            )
    return predictions, loss_total / sample_count if sample_count else 0.0


def best_metric(row: MetricRow) -> float:
    return (
        0.50 * _metric_float(row, "Full Support@5")
        + 0.30 * _metric_float(row, "Recall@5")
        + 0.20 * _metric_float(row, "MRR")
    )


def _metric_float(row: MetricRow, key: str) -> float:
    value = row[key]
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"Metric column must be numeric for best checkpoint selection: {key}"
        )
    return float(value)


def _score_actions(
    model: EvidenceScoringModel,
    encoded,
    task_index: int,
    selected: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = model.score_decoder_actions(
        encoded,
        task_index=task_index,
        selected_local_indices=selected,
    )
    return scores.candidate_logits, scores.stop_logit
