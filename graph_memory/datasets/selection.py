from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, NoReturn, cast

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.ranking import RankedResult
from graph_memory.datasets.hotpotqa.projectors import (
    HotpotQAToEvidenceEvaluationRequest,
    HotpotQAToGraphBuildRequest,
    HotpotQAToTemporalMemoryRankingRequest,
    HotpotQAToTextRankingRequest,
)
from graph_memory.datasets.hotpotqa.records import HotpotQALabelRecord, HotpotQARankingRecord
from graph_memory.datasets.longmemeval.projectors import (
    LongMemEvalToEvidenceEvaluationRequest,
    LongMemEvalToGraphBuildRequest,
    LongMemEvalToTemporalMemoryRankingRequest,
    LongMemEvalToTextRankingRequest,
)
from graph_memory.datasets.longmemeval.records import LongMemEvalLabelRecord, LongMemEvalRankingRecord
from graph_memory.datasets.twowiki.projectors import (
    TwoWikiToEvidenceEvaluationRequest,
    TwoWikiToGraphBuildRequest,
    TwoWikiToTemporalMemoryRankingRequest,
    TwoWikiToTextRankingRequest,
)
from graph_memory.datasets.twowiki.records import TwoWikiLabelRecord, TwoWikiRankingRecord
from graph_memory.evaluation.requests import EvidenceEvaluationRequest, EvidenceLabel
from graph_memory.graphs.requests import GraphBuildRequest
from graph_memory.retrieval.requests import TemporalMemoryRankingRequest, TextRankingRequest
from graph_memory.validation import (
    validate_hotpotqa_label_records,
    validate_hotpotqa_ranking_records,
    validate_longmemeval_label_records,
    validate_longmemeval_ranking_records,
    validate_twowiki_label_records,
    validate_twowiki_ranking_records,
)

DatasetId = Literal["hotpotqa", "twowiki", "longmemeval"]


def validate_ranking_records_for_dataset(dataset: DatasetId, records: object) -> None:
    if dataset == "hotpotqa":
        validate_hotpotqa_ranking_records(records)
        return
    if dataset == "twowiki":
        validate_twowiki_ranking_records(records)
        return
    if dataset == "longmemeval":
        validate_longmemeval_ranking_records(records)
        return
    _unsupported_dataset(dataset)


def validate_label_records_for_dataset(dataset: DatasetId, labels: object, records_by_task_id: object) -> None:
    if dataset == "hotpotqa":
        validate_hotpotqa_label_records(labels, records_by_task_id)
        return
    if dataset == "twowiki":
        validate_twowiki_label_records(labels, records_by_task_id)
        return
    if dataset == "longmemeval":
        validate_longmemeval_label_records(labels, records_by_task_id)
        return
    _unsupported_dataset(dataset)


def text_ranking_requests_for_dataset(dataset: DatasetId, records: Sequence[object]) -> list[TextRankingRequest]:
    if dataset == "hotpotqa":
        projector = HotpotQAToTextRankingRequest()
        return [projector.project(cast(HotpotQARankingRecord, record)) for record in records]
    if dataset == "twowiki":
        projector = TwoWikiToTextRankingRequest()
        return [projector.project(cast(TwoWikiRankingRecord, record)) for record in records]
    if dataset == "longmemeval":
        projector = LongMemEvalToTextRankingRequest()
        return [projector.project(cast(LongMemEvalRankingRecord, record)) for record in records]
    _unsupported_dataset(dataset)


def temporal_memory_requests_for_dataset(
    dataset: DatasetId,
    records: Sequence[object],
    importance_by_item_id: Mapping[str, float] | None = None,
) -> list[TemporalMemoryRankingRequest]:
    importance = importance_by_item_id or {}
    if dataset == "hotpotqa":
        projector = HotpotQAToTemporalMemoryRankingRequest()
        return [projector.project(cast(HotpotQARankingRecord, record), importance) for record in records]
    if dataset == "twowiki":
        projector = TwoWikiToTemporalMemoryRankingRequest()
        return [projector.project(cast(TwoWikiRankingRecord, record), importance) for record in records]
    if dataset == "longmemeval":
        projector = LongMemEvalToTemporalMemoryRankingRequest()
        return [projector.project(cast(LongMemEvalRankingRecord, record), importance) for record in records]
    _unsupported_dataset(dataset)


def graph_build_requests_for_dataset(dataset: DatasetId, records: Sequence[object]) -> list[GraphBuildRequest]:
    if dataset == "hotpotqa":
        projector = HotpotQAToGraphBuildRequest()
        return [projector.project(cast(HotpotQARankingRecord, record)) for record in records]
    if dataset == "twowiki":
        projector = TwoWikiToGraphBuildRequest()
        return [projector.project(cast(TwoWikiRankingRecord, record)) for record in records]
    if dataset == "longmemeval":
        projector = LongMemEvalToGraphBuildRequest()
        return [projector.project(cast(LongMemEvalRankingRecord, record)) for record in records]
    _unsupported_dataset(dataset)


def evidence_evaluation_request_for_dataset(
    dataset: DatasetId,
    *,
    predictions: Sequence[RankedResult],
    labels: Sequence[object],
    graphs: Sequence[MemoryGraph],
) -> EvidenceEvaluationRequest:
    if dataset == "hotpotqa":
        return HotpotQAToEvidenceEvaluationRequest().project(
            predictions=predictions,
            labels=cast(Sequence[HotpotQALabelRecord], labels),
            graphs=graphs,
        )
    if dataset == "twowiki":
        return TwoWikiToEvidenceEvaluationRequest().project(
            predictions=predictions,
            labels=cast(Sequence[TwoWikiLabelRecord], labels),
            graphs=graphs,
        )
    if dataset == "longmemeval":
        return LongMemEvalToEvidenceEvaluationRequest().project(
            predictions=predictions,
            labels=cast(Sequence[LongMemEvalLabelRecord], labels),
            graphs=graphs,
        )
    _unsupported_dataset(dataset)


def evidence_labels_for_dataset(dataset: DatasetId, labels: Sequence[object]) -> list[EvidenceLabel]:
    request = evidence_evaluation_request_for_dataset(dataset, predictions=[], labels=labels, graphs=[])
    return list(request.labels)


def _unsupported_dataset(dataset: object) -> NoReturn:
    raise ValueError(f"Unsupported dataset: {dataset!r}.")


__all__ = [
    "DatasetId",
    "evidence_evaluation_request_for_dataset",
    "evidence_labels_for_dataset",
    "graph_build_requests_for_dataset",
    "temporal_memory_requests_for_dataset",
    "text_ranking_requests_for_dataset",
    "validate_label_records_for_dataset",
    "validate_ranking_records_for_dataset",
]
