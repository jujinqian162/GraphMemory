from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from graph_memory.contracts.common import NodeId, TaskId, TrainPairSampleType
from graph_memory.contracts.training_pairs import TrainPairRecord
from graph_memory.embeddings import format_dense_passage, format_dense_query
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.models.dense_finetune.contracts import (
    DenseFinetuneDataSettings,
    DenseFinetuneDatasetBuildResult,
    DenseFinetuneExample,
    DenseFinetuneIREvaluatorPayload,
)
from graph_memory.retrieval.requests import TextCandidate, TextRankingRequest

_NEGATIVE_PRIORITY: dict[TrainPairSampleType, int] = {
    "hard_dense": 0,
    "hard_bm25": 1,
    "hard_graph_neighbor": 2,
    "easy_random": 3,
    "positive": 4,
}


def build_dense_finetune_examples(
    *,
    ranking_requests: Sequence[TextRankingRequest],
    train_pairs: Sequence[TrainPairRecord],
    settings: DenseFinetuneDataSettings,
    query_prefix: str = "query: ",
    passage_prefix: str = "passage: ",
) -> DenseFinetuneDatasetBuildResult:
    request_by_id = _text_request_by_id(ranking_requests)
    candidate_by_task = {
        task_id: {candidate.item_id: candidate for candidate in request.candidates}
        for task_id, request in request_by_id.items()
    }
    indexed_pairs_by_task: dict[TaskId, list[tuple[int, TrainPairRecord]]] = defaultdict(list)
    for index, pair in enumerate(train_pairs):
        task_id = pair["task_id"]
        if task_id not in request_by_id:
            raise ValueError(f"Unknown train pair task_id={task_id}.")
        _require_candidate(candidate_by_task[task_id], task_id=task_id, node_id=pair["node_id"])
        indexed_pairs_by_task[task_id].append((index, pair))

    examples: list[DenseFinetuneExample] = []
    emitted: set[tuple[TaskId, NodeId, NodeId | None]] = set()
    for request in request_by_id.values():
        task_id = request.task_id
        indexed_pairs = indexed_pairs_by_task.get(task_id, [])
        positives = [pair for _, pair in indexed_pairs if pair["label"] == 1]
        negatives = [
            pair
            for _, pair in sorted(
                ((index, pair) for index, pair in indexed_pairs if pair["label"] == 0),
                key=lambda indexed_pair: (_NEGATIVE_PRIORITY[indexed_pair[1]["sample_type"]], indexed_pair[0]),
            )
        ]
        selected_negatives = negatives[: settings.hard_negatives_per_positive]

        for positive_pair in positives:
            positive_item = _require_candidate(
                candidate_by_task[task_id],
                task_id=task_id,
                node_id=positive_pair["node_id"],
            )
            if not selected_negatives:
                example = _build_example(
                    request,
                    positive_item=positive_item,
                    negative_item=None,
                    positive_node_id=positive_pair["node_id"],
                    negative_sample_type=None,
                    query_prefix=query_prefix,
                    passage_prefix=passage_prefix,
                )
                key = (example.task_id, example.positive_node_id, example.negative_node_id)
                if key not in emitted:
                    examples.append(example)
                    emitted.add(key)
                continue

            for negative_pair in selected_negatives:
                negative_item = _require_candidate(
                    candidate_by_task[task_id],
                    task_id=task_id,
                    node_id=negative_pair["node_id"],
                )
                example = _build_example(
                    request,
                    positive_item=positive_item,
                    negative_item=negative_item,
                    positive_node_id=positive_pair["node_id"],
                    negative_sample_type=negative_pair["sample_type"],
                    query_prefix=query_prefix,
                    passage_prefix=passage_prefix,
                )
                key = (example.task_id, example.positive_node_id, example.negative_node_id)
                if key not in emitted:
                    examples.append(example)
                    emitted.add(key)

    return DenseFinetuneDatasetBuildResult(
        examples=tuple(examples),
        rows=tuple(_row_from_example(example) for example in examples),
    )


def build_ir_evaluator_payload(
    *,
    ranking_requests: Sequence[TextRankingRequest],
    labels: Sequence[EvidenceLabel],
    query_prefix: str = "query: ",
    passage_prefix: str = "passage: ",
) -> DenseFinetuneIREvaluatorPayload:
    labels_by_task_id: dict[TaskId, EvidenceLabel] = {}
    for label in labels:
        task_id = label.task_id
        if task_id in labels_by_task_id:
            raise ValueError(f"Duplicate task labels for task_id={task_id}.")
        labels_by_task_id[task_id] = label

    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}
    relevant_docs: dict[str, set[str]] = {}
    for request in _text_request_by_id(ranking_requests).values():
        task_id = request.task_id
        label = labels_by_task_id.get(task_id)
        if label is None:
            raise ValueError(f"Missing task labels for task_id={task_id}.")
        candidate_by_node_id = {candidate.item_id: candidate for candidate in request.candidates}
        queries[task_id] = format_dense_query(request, query_prefix=query_prefix)
        for candidate in request.candidates:
            corpus[_corpus_id(task_id, candidate.item_id)] = format_dense_passage(
                candidate,
                passage_prefix=passage_prefix,
            )
        relevant_docs[task_id] = set()
        for node_id in label.gold_evidence_item_ids:
            _require_candidate(candidate_by_node_id, task_id=task_id, node_id=node_id)
            relevant_docs[task_id].add(_corpus_id(task_id, node_id))

    return DenseFinetuneIREvaluatorPayload(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
    )


def _text_request_by_id(ranking_requests: Sequence[TextRankingRequest]) -> dict[TaskId, TextRankingRequest]:
    request_by_id: dict[TaskId, TextRankingRequest] = {}
    for request in ranking_requests:
        task_id = request.task_id
        if task_id in request_by_id:
            raise ValueError(f"Duplicate task input for task_id={task_id}.")
        request_by_id[task_id] = request
    return request_by_id


def _require_candidate(
    candidate_by_node_id: dict[NodeId, TextCandidate],
    *,
    task_id: TaskId,
    node_id: NodeId,
) -> TextCandidate:
    try:
        return candidate_by_node_id[node_id]
    except KeyError as error:
        raise ValueError(f"Unknown dense-ft pair node for task_id={task_id} node_id={node_id}.") from error


def _build_example(
    request: TextRankingRequest,
    *,
    positive_item: TextCandidate,
    negative_item: TextCandidate | None,
    positive_node_id: NodeId,
    negative_sample_type: TrainPairSampleType | None,
    query_prefix: str,
    passage_prefix: str,
) -> DenseFinetuneExample:
    return DenseFinetuneExample(
        task_id=request.task_id,
        positive_node_id=positive_node_id,
        negative_node_id=None if negative_item is None else negative_item.item_id,
        anchor=format_dense_query(request, query_prefix=query_prefix),
        positive=format_dense_passage(positive_item, passage_prefix=passage_prefix),
        negative=None if negative_item is None else format_dense_passage(negative_item, passage_prefix=passage_prefix),
        negative_sample_type=negative_sample_type,
    )


def _row_from_example(example: DenseFinetuneExample) -> dict[str, str]:
    row = {"anchor": example.anchor, "positive": example.positive}
    if example.negative is not None:
        row["negative"] = example.negative
    return row


def _corpus_id(task_id: TaskId, node_id: NodeId) -> str:
    return f"{task_id}::{node_id}"
