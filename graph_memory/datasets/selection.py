from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, NoReturn, TypeAlias

from pydantic import TypeAdapter

from graph_memory.datasets.hotpotqa.projectors import (
    HotpotQAToEvidenceEvaluationRequest,
    HotpotQAToEvidenceGraphBuildRequest,
    HotpotQAToTextRankingRequest,
)
from graph_memory.datasets.hotpotqa.records import (
    HotpotQALabelRecord,
    HotpotQARankingRecord,
)
from graph_memory.datasets.isetrace.benchmark_records import (
    ISETraceLabelRecord,
    ISETraceRankingRecord,
)
from graph_memory.datasets.musique.projectors import (
    MuSiQueToEvidenceEvaluationRequest,
    MuSiQueToEvidenceGraphBuildRequest,
    MuSiQueToTextRankingRequest,
)
from graph_memory.datasets.musique.records import (
    MuSiQueLabelRecord,
    MuSiQueRankingRecord,
)
from graph_memory.datasets.twowiki.projectors import (
    TwoWikiToEvidenceEvaluationRequest,
    TwoWikiToEvidenceGraphBuildRequest,
    TwoWikiToTextRankingRequest,
)
from graph_memory.datasets.twowiki.records import (
    TwoWikiLabelRecord,
    TwoWikiRankingRecord,
)
from graph_memory.evaluation.requests import (
    EvidenceEvaluationRequest,
    EvidenceLabel,
    SpanEvidenceEvaluationRequest,
    SpanEvidenceLabel,
)
from graph_memory.graphs.contracts import EvidenceGraph
from graph_memory.graphs.requests import EvidenceGraphBuildRequest
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.retrieval.results import RankedResult

DatasetId = Literal[
    "hotpotqa",
    "twowiki",
    "musique",
    "isetrace",
]
DatasetRankingRecord: TypeAlias = (
    HotpotQARankingRecord
    | TwoWikiRankingRecord
    | MuSiQueRankingRecord
    | ISETraceRankingRecord
)
DatasetLabelRecord: TypeAlias = (
    HotpotQALabelRecord
    | TwoWikiLabelRecord
    | MuSiQueLabelRecord
    | ISETraceLabelRecord
)

_HOTPOT_RANKINGS = TypeAdapter(list[HotpotQARankingRecord])
_HOTPOT_LABELS = TypeAdapter(list[HotpotQALabelRecord])
_TWOWIKI_RANKINGS = TypeAdapter(list[TwoWikiRankingRecord])
_TWOWIKI_LABELS = TypeAdapter(list[TwoWikiLabelRecord])
_MUSIQUE_RANKINGS = TypeAdapter(list[MuSiQueRankingRecord])
_MUSIQUE_LABELS = TypeAdapter(list[MuSiQueLabelRecord])
_ISETRACE_RANKINGS = TypeAdapter(list[ISETraceRankingRecord])
_ISETRACE_LABELS = TypeAdapter(list[ISETraceLabelRecord])


def ranking_records_for_dataset(
    dataset: DatasetId, records: object
) -> list[DatasetRankingRecord]:
    if dataset == "hotpotqa":
        return list(_HOTPOT_RANKINGS.validate_python(records))
    if dataset == "twowiki":
        return list(_TWOWIKI_RANKINGS.validate_python(records))
    if dataset == "musique":
        return list(_MUSIQUE_RANKINGS.validate_python(records))
    if dataset == "isetrace":
        return list(_ISETRACE_RANKINGS.validate_python(records))
    _unsupported_dataset(dataset)


def label_records_for_dataset(
    dataset: DatasetId, labels: object
) -> list[DatasetLabelRecord]:
    if dataset == "hotpotqa":
        return list(_HOTPOT_LABELS.validate_python(labels))
    if dataset == "twowiki":
        return list(_TWOWIKI_LABELS.validate_python(labels))
    if dataset == "musique":
        return list(_MUSIQUE_LABELS.validate_python(labels))
    if dataset == "isetrace":
        return list(_ISETRACE_LABELS.validate_python(labels))
    _unsupported_dataset(dataset)


def text_ranking_requests_for_dataset(
    dataset: DatasetId,
    records: Sequence[object],
    *,
    isetrace_representation: Literal["flat", "provenance"] = "flat",
) -> list[TextRankingRequest]:
    validated = ranking_records_for_dataset(dataset, records)
    if dataset == "hotpotqa":
        projector = HotpotQAToTextRankingRequest()
        return [projector.project(record) for record in validated]
    if dataset == "twowiki":
        projector = TwoWikiToTextRankingRequest()
        return [projector.project(record) for record in validated]
    if dataset == "musique":
        projector = MuSiQueToTextRankingRequest()
        return [projector.project(record) for record in validated]
    if dataset == "isetrace":
        candidates_attr = (
            "provenance_candidates"
            if isetrace_representation == "provenance"
            else "flat_candidates"
        )
        return [
            TextRankingRequest(
                task_id=record.task_id,
                query_text=record.query_text,
                candidates=getattr(record, candidates_attr),
            )
            for record in _ISETRACE_RANKINGS.validate_python(validated)
        ]
    _unsupported_dataset(dataset)


def evidence_graph_build_requests_for_dataset(
    dataset: DatasetId, records: Sequence[object]
) -> list[EvidenceGraphBuildRequest]:
    validated = ranking_records_for_dataset(dataset, records)
    if dataset == "hotpotqa":
        projector = HotpotQAToEvidenceGraphBuildRequest()
        return [projector.project(record) for record in validated]
    if dataset == "twowiki":
        projector = TwoWikiToEvidenceGraphBuildRequest()
        return [projector.project(record) for record in validated]
    if dataset == "musique":
        projector = MuSiQueToEvidenceGraphBuildRequest()
        return [projector.project(record) for record in validated]
    if dataset == "isetrace":
        raise TypeError(
            "isetrace uses method-native retrieval views and span evaluation; "
            "legacy EvidenceGraph projection is unsupported"
        )
    _unsupported_dataset(dataset)


def evidence_evaluation_request_for_dataset(
    dataset: DatasetId,
    *,
    predictions: Sequence[RankedResult],
    labels: Sequence[object],
    graphs: Sequence[EvidenceGraph],
) -> EvidenceEvaluationRequest | SpanEvidenceEvaluationRequest:
    validated_labels = label_records_for_dataset(dataset, labels)
    if dataset == "hotpotqa":
        return HotpotQAToEvidenceEvaluationRequest().project(
            predictions=predictions,
            labels=_HOTPOT_LABELS.validate_python(validated_labels),
            graphs=graphs,
        )
    if dataset == "twowiki":
        return TwoWikiToEvidenceEvaluationRequest().project(
            predictions=predictions,
            labels=_TWOWIKI_LABELS.validate_python(validated_labels),
            graphs=graphs,
        )
    if dataset == "musique":
        return MuSiQueToEvidenceEvaluationRequest().project(
            predictions=predictions,
            labels=_MUSIQUE_LABELS.validate_python(validated_labels),
            graphs=graphs,
        )
    if dataset == "isetrace":
        return SpanEvidenceEvaluationRequest(
            predictions=tuple(predictions),
            labels=tuple(
                SpanEvidenceLabel(
                    task_id=label.task_id,
                    gold_evidence_spans=label.gold_evidence_spans,
                )
                for label in _ISETRACE_LABELS.validate_python(validated_labels)
            ),
        )
    _unsupported_dataset(dataset)


def evidence_labels_for_dataset(
    dataset: DatasetId, labels: Sequence[object]
) -> list[EvidenceLabel]:
    if dataset == "isetrace":
        raise TypeError(
            "isetrace span labels are evaluation-only; training methods are unsupported"
        )
    request = evidence_evaluation_request_for_dataset(
        dataset, predictions=(), labels=labels, graphs=()
    )
    if not isinstance(request, EvidenceEvaluationRequest):
        raise TypeError(f"dataset={dataset!r} does not provide node evidence labels")
    return list(request.labels)


def _unsupported_dataset(dataset: object) -> NoReturn:
    raise ValueError(f"Unsupported dataset: {dataset!r}.")


__all__ = [
    "DatasetId",
    "DatasetLabelRecord",
    "DatasetRankingRecord",
    "evidence_evaluation_request_for_dataset",
    "evidence_graph_build_requests_for_dataset",
    "evidence_labels_for_dataset",
    "label_records_for_dataset",
    "ranking_records_for_dataset",
    "text_ranking_requests_for_dataset",
]
