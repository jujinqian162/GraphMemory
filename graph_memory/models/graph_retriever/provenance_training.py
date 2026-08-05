from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, TypeAlias

import torch
import torch.nn.functional as F

from graph_memory.evaluation.contracts import MetricRow
from graph_memory.evaluation.requests import EvidenceEvaluationRequest, EvidenceLabel
from graph_memory.evaluation.service import evaluate_results
from graph_memory.models.graph_retriever.batching import (
    move_training_batch,
    split_batch_node_scores,
)
from graph_memory.models.graph_retriever.config.records import (
    RgcnModelConfig,
    RgcnTrainingConfig,
)
from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
from graph_memory.models.graph_retriever.internals.contracts import TrainingBatch
from graph_memory.models.graph_retriever.internals.neural import EvidenceScoringModel
from graph_memory.models.graph_retriever.provenance import (
    tensorize_provenance_dev_task,
    tensorize_provenance_training_task,
)
from graph_memory.models.graph_retriever.selection import (
    RgcnSelectionSettings,
    build_selection_metrics,
)
from graph_memory.models.graph_retriever.training import (
    CheckpointCallback,
    RgcnDevEpochEvaluation,
    RgcnTrainingResult,
    train_materialized_graph_retriever,
)
from graph_memory.retrieval.contracts import RankedNode
from graph_memory.retrieval.execution.results import assemble_ranked_result
from graph_memory.retrieval.methods.ids import RetrievalMethodId
from graph_memory.retrieval.requests import ProvenanceRgcnRequest, TextRankingRequest
from graph_memory.retrieval.results import RankedResult
from graph_memory.training_pairs.contracts import TrainPairDataset, TrainPairRecord

QueryOrigin: TypeAlias = Literal["natural", "template"]


def train_provenance_graph_retriever(
    *,
    train_requests: Sequence[ProvenanceRgcnRequest],
    train_labels: Sequence[EvidenceLabel],
    train_pairs: Sequence[TrainPairRecord],
    dev_requests: Sequence[ProvenanceRgcnRequest],
    dev_labels: Sequence[EvidenceLabel],
    dev_query_origins: Mapping[str, QueryOrigin],
    model_config: RgcnModelConfig,
    training_config: RgcnTrainingConfig,
    text_embedding_provider: TextEmbeddingProvider,
    dev_text_embedding_provider: TextEmbeddingProvider | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
    device: str | torch.device,
) -> RgcnTrainingResult:
    """Adapt provenance tasks to the shared R-GCN optimizer and dev loop."""

    if model_config.method_name != RetrievalMethodId.PROVENANCE_RGCN:
        raise ValueError("provenance training requires method_name=provenance_rgcn")
    train_request_list = list(train_requests)
    train_label_list = list(train_labels)
    train_pair_list = list(train_pairs)
    dev_request_list = list(dev_requests)
    dev_label_list = list(dev_labels)
    _validate_request_label_alignment(train_request_list, train_label_list, "train")
    _validate_request_label_alignment(dev_request_list, dev_label_list, "dev")
    expected_dev_ids = {request.task_id for request in dev_request_list}
    if set(dev_query_origins) != expected_dev_ids:
        raise ValueError("dev query origins must align exactly with dev task IDs")
    selection_origin: QueryOrigin = (
        "natural"
        if any(origin == "natural" for origin in dev_query_origins.values())
        else "template"
    )

    text_train_requests = [_as_text_request(request) for request in train_request_list]
    validated_pairs = TrainPairDataset(
        requests=tuple(text_train_requests),
        labels=tuple(train_label_list),
        graphs=(),
        pairs=tuple(train_pair_list),
    )
    pairs_by_task_id: dict[str, list[TrainPairRecord]] = defaultdict(list)
    for pair in validated_pairs.pairs:
        pairs_by_task_id[pair.task_id].append(pair)
    train_tasks = [
        tensorize_provenance_training_task(
            request,
            pairs_by_task_id[request.task_id],
            model_config=model_config,
            text_embedding_provider=text_embedding_provider,
        )
        for request in train_request_list
    ]
    dev_labels_by_id = {label.task_id: label for label in dev_label_list}
    dev_tasks = [
        tensorize_provenance_dev_task(
            request,
            dev_labels_by_id[request.task_id],
            model_config=model_config,
            text_embedding_provider=(
                dev_text_embedding_provider or text_embedding_provider
            ),
        )
        for request in dev_request_list
    ]

    def evaluate_dev_epoch(
        model: EvidenceScoringModel,
        batches: Iterable[TrainingBatch],
        eval_device: torch.device,
    ) -> RgcnDevEpochEvaluation:
        predictions, dev_loss = _predict_provenance_dev_from_batches(
            model=model,
            requests=dev_request_list,
            batches=batches,
            device=eval_device,
        )
        rows_by_origin = _evaluate_by_origin(
            predictions=predictions,
            labels=dev_label_list,
            query_origins=dev_query_origins,
        )
        overall = rows_by_origin["overall"]
        selected = rows_by_origin[selection_origin]
        metric_records: dict[str, object] = {
            "dev_query_selection_origin": selection_origin,
            "dev_recall_at_5": overall.recall_at_5,
            "dev_full_support_at_5": overall.full_support_at_5,
            "dev_full_support_at_10": overall.full_support_at_10,
            "dev_mrr": overall.mrr,
        }
        for origin in ("natural", "template"):
            origin_row = rows_by_origin.get(origin)
            if origin_row is None:
                continue
            metric_records.update(
                {
                    f"dev_{origin}_recall_at_5": origin_row.recall_at_5,
                    f"dev_{origin}_full_support_at_5": origin_row.full_support_at_5,
                    f"dev_{origin}_full_support_at_10": origin_row.full_support_at_10,
                    f"dev_{origin}_mrr": origin_row.mrr,
                }
            )
        return RgcnDevEpochEvaluation(
            dev_loss=dev_loss,
            selection_metrics=build_selection_metrics(
                dev_full_support_at_5=selected.full_support_at_5,
                dev_full_support_at_10=selected.full_support_at_10,
                dev_recall_at_5=selected.recall_at_5,
                dev_mrr=selected.mrr,
                dev_loss=dev_loss,
            ),
            metric_records=metric_records,
        )

    return train_materialized_graph_retriever(
        train_tasks=train_tasks,
        dev_tasks=dev_tasks,
        train_pairs=train_pair_list,
        model_config=model_config,
        training_config=training_config,
        dev_evaluation_callback=evaluate_dev_epoch,
        selection_settings=RgcnSelectionSettings(
            best_metric="dev_recall_at_5", higher_is_better=True
        ),
        checkpoint_callback=checkpoint_callback,
        device=device,
        progress_desc="provenance-rgcn epochs",
    )


def _predict_provenance_dev_from_batches(
    *,
    model: EvidenceScoringModel,
    requests: Sequence[ProvenanceRgcnRequest],
    batches: Iterable[TrainingBatch],
    device: torch.device,
) -> tuple[list[RankedResult], float]:
    scores_by_task_id: dict[str, list[RankedNode]] = defaultdict(list)
    loss_total = 0.0
    sample_count = 0
    model.eval()
    with torch.no_grad():
        for batch in batches:
            moved_batch = move_training_batch(batch, device)
            logits = model(moved_batch)
            loss_sum = F.binary_cross_entropy_with_logits(
                logits, moved_batch.labels, reduction="sum"
            )
            loss_total += float(loss_sum.detach().cpu())
            sample_count += int(moved_batch.labels.shape[0])
            for task_id, rows in split_batch_node_scores(batch, logits).items():
                scores_by_task_id[task_id].extend(
                    RankedNode(node_id=node_id, score=score)
                    for node_id, score in rows
                )
    predictions = [
        assemble_ranked_result(
            request=request,
            method=RetrievalMethodId.PROVENANCE_RGCN,
            ranked_nodes=sorted(
                scores_by_task_id[request.task_id],
                key=lambda row: (-row.score, row.node_id),
            ),
            top_k=10,
            latency_ms=0.0,
            retrieved_edges=(),
            native_trace=None,
        )
        for request in requests
    ]
    return predictions, loss_total / sample_count if sample_count else 0.0


def _evaluate_by_origin(
    *,
    predictions: Sequence[RankedResult],
    labels: Sequence[EvidenceLabel],
    query_origins: Mapping[str, QueryOrigin],
) -> dict[str, MetricRow]:
    predictions_by_id = {prediction.task_id: prediction for prediction in predictions}
    labels_by_id = {label.task_id: label for label in labels}

    def evaluate(task_ids: set[str]) -> MetricRow:
        return evaluate_results(
            EvidenceEvaluationRequest(
                predictions=tuple(
                    predictions_by_id[task_id] for task_id in sorted(task_ids)
                ),
                labels=tuple(labels_by_id[task_id] for task_id in sorted(task_ids)),
                graphs=(),
            )
        )[0]

    rows = {"overall": evaluate(set(labels_by_id))}
    for origin in ("natural", "template"):
        task_ids = {
            task_id
            for task_id, task_origin in query_origins.items()
            if task_origin == origin
        }
        if task_ids:
            rows[origin] = evaluate(task_ids)
    return rows


def _validate_request_label_alignment(
    requests: Sequence[ProvenanceRgcnRequest],
    labels: Sequence[EvidenceLabel],
    split: str,
) -> None:
    request_ids = [request.task_id for request in requests]
    label_ids = [label.task_id for label in labels]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError(f"provenance {split} request task IDs must be unique")
    if len(label_ids) != len(set(label_ids)):
        raise ValueError(f"provenance {split} label task IDs must be unique")
    if set(request_ids) != set(label_ids):
        raise ValueError(f"provenance {split} requests and labels must align")
    requests_by_id = {request.task_id: request for request in requests}
    for label in labels:
        candidate_ids = {
            candidate.item_id for candidate in requests_by_id[label.task_id].candidates
        }
        missing = set(label.gold_evidence_item_ids) - candidate_ids
        if missing:
            raise ValueError(
                f"provenance {split} positives are not candidates: {sorted(missing)}"
            )


def _as_text_request(request: ProvenanceRgcnRequest) -> TextRankingRequest:
    return TextRankingRequest(
        task_id=request.task_id,
        query_text=request.query_text,
        candidates=request.candidates,
    )


__all__ = ["QueryOrigin", "train_provenance_graph_retriever"]
