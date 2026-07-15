from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, NoReturn, cast

from graph_memory.contracts.graphs import EvidenceGraph
from graph_memory.contracts.ranking import RankedResult
from graph_memory.datasets.hotpotqa.projectors import (
    HotpotQAToEvidenceEvaluationRequest,
    HotpotQAToEvidenceGraphBuildRequest,
    HotpotQAToTextRankingRequest,
)
from graph_memory.datasets.hotpotqa.records import (
    HotpotQALabelRecord,
    HotpotQARankingRecord,
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
from graph_memory.datasets.traject_bench.projectors import (
    TrajectBenchToEvidenceEvaluationRequest,
    TrajectBenchToEvidenceGraphBuildRequest,
    TrajectBenchToTextRankingRequest,
)
from graph_memory.datasets.traject_bench.records import (
    TrajectBenchLabelRecord,
    TrajectBenchRankingRecord,
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
from graph_memory.evaluation.requests import EvidenceEvaluationRequest, EvidenceLabel
from graph_memory.graphs.requests import EvidenceGraphBuildRequest
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.validation import (
    validate_hotpotqa_label_records,
    validate_hotpotqa_ranking_records,
    validate_musique_label_records,
    validate_musique_ranking_records,
    validate_traject_bench_label_records,
    validate_traject_bench_ranking_records,
    validate_twowiki_label_records,
    validate_twowiki_ranking_records,
)

DatasetId = Literal["hotpotqa", "twowiki", "musique", "traject_bench"]


def validate_ranking_records_for_dataset(dataset: DatasetId, records: object) -> None:
    if dataset == "hotpotqa":
        validate_hotpotqa_ranking_records(records)
        return
    if dataset == "twowiki":
        validate_twowiki_ranking_records(records)
        return
    if dataset == "musique":
        validate_musique_ranking_records(records)
        return
    if dataset == "traject_bench":
        validate_traject_bench_ranking_records(records)
        return
    _unsupported_dataset(dataset)


def validate_label_records_for_dataset(
    dataset: DatasetId, labels: object, records_by_task_id: object
) -> None:
    if dataset == "hotpotqa":
        validate_hotpotqa_label_records(labels, records_by_task_id)
        return
    if dataset == "twowiki":
        validate_twowiki_label_records(labels, records_by_task_id)
        return
    if dataset == "musique":
        validate_musique_label_records(labels, records_by_task_id)
        return
    if dataset == "traject_bench":
        validate_traject_bench_label_records(labels, records_by_task_id)
        return
    _unsupported_dataset(dataset)


def text_ranking_requests_for_dataset(
    dataset: DatasetId, records: Sequence[object]
) -> list[TextRankingRequest]:
    if dataset == "hotpotqa":
        projector = HotpotQAToTextRankingRequest()
        return [
            projector.project(cast(HotpotQARankingRecord, record)) for record in records
        ]
    if dataset == "twowiki":
        projector = TwoWikiToTextRankingRequest()
        return [
            projector.project(cast(TwoWikiRankingRecord, record)) for record in records
        ]
    if dataset == "musique":
        projector = MuSiQueToTextRankingRequest()
        return [
            projector.project(cast(MuSiQueRankingRecord, record)) for record in records
        ]
    if dataset == "traject_bench":
        projector = TrajectBenchToTextRankingRequest()
        return [
            projector.project(cast(TrajectBenchRankingRecord, record))
            for record in records
        ]
    _unsupported_dataset(dataset)


def evidence_graph_build_requests_for_dataset(
    dataset: DatasetId, records: Sequence[object]
) -> list[EvidenceGraphBuildRequest]:
    if dataset == "hotpotqa":
        projector = HotpotQAToEvidenceGraphBuildRequest()
        return [
            projector.project(cast(HotpotQARankingRecord, record)) for record in records
        ]
    if dataset == "twowiki":
        projector = TwoWikiToEvidenceGraphBuildRequest()
        return [
            projector.project(cast(TwoWikiRankingRecord, record)) for record in records
        ]
    if dataset == "musique":
        projector = MuSiQueToEvidenceGraphBuildRequest()
        return [
            projector.project(cast(MuSiQueRankingRecord, record)) for record in records
        ]
    if dataset == "traject_bench":
        projector = TrajectBenchToEvidenceGraphBuildRequest()
        return [
            projector.project(cast(TrajectBenchRankingRecord, record))
            for record in records
        ]
    _unsupported_dataset(dataset)


def evidence_evaluation_request_for_dataset(
    dataset: DatasetId,
    *,
    predictions: Sequence[RankedResult],
    labels: Sequence[object],
    graphs: Sequence[EvidenceGraph],
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
    if dataset == "musique":
        return MuSiQueToEvidenceEvaluationRequest().project(
            predictions=predictions,
            labels=cast(Sequence[MuSiQueLabelRecord], labels),
            graphs=graphs,
        )
    if dataset == "traject_bench":
        return TrajectBenchToEvidenceEvaluationRequest().project(
            predictions=predictions,
            labels=cast(Sequence[TrajectBenchLabelRecord], labels),
            graphs=graphs,
        )
    _unsupported_dataset(dataset)


def evidence_labels_for_dataset(
    dataset: DatasetId, labels: Sequence[object]
) -> list[EvidenceLabel]:
    request = evidence_evaluation_request_for_dataset(
        dataset, predictions=[], labels=labels, graphs=[]
    )
    return list(request.labels)


def _unsupported_dataset(dataset: object) -> NoReturn:
    raise ValueError(f"Unsupported dataset: {dataset!r}.")


__all__ = [
    "DatasetId",
    "evidence_evaluation_request_for_dataset",
    "evidence_labels_for_dataset",
    "evidence_graph_build_requests_for_dataset",
    "text_ranking_requests_for_dataset",
    "validate_label_records_for_dataset",
    "validate_ranking_records_for_dataset",
]
