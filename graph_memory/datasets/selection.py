from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, NoReturn, TypeAlias

from pydantic import TypeAdapter

from graph_memory.datasets.hotpotqa.records import (
    HotpotQALabelRecord,
    HotpotQARankingRecord,
)
from graph_memory.datasets.isetrace.benchmark_records import (
    ISETraceLabelRecord,
    ISETraceRankingRecord,
)
from graph_memory.datasets.musique.records import (
    MuSiQueLabelRecord,
    MuSiQueRankingRecord,
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
from graph_memory.graphs.requests import (
    EvidenceGraphBuildNode,
    EvidenceGraphBuildRequest,
)
from graph_memory.retrieval.requests import TextCandidate, TextRankingRequest
from graph_memory.retrieval.results import RankedResult

DatasetId = Literal["hotpotqa", "twowiki", "musique", "isetrace"]
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
    dataset: DatasetId,
    records: object,
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
    dataset: DatasetId,
    labels: object,
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
    if dataset == "hotpotqa":
        return [
            _sentence_text_request(record)
            for record in _HOTPOT_RANKINGS.validate_python(records)
        ]
    if dataset == "twowiki":
        return [
            _sentence_text_request(record)
            for record in _TWOWIKI_RANKINGS.validate_python(records)
        ]
    if dataset == "musique":
        return [
            TextRankingRequest(
                task_id=record.task_id,
                query_text=record.question,
                candidates=tuple(
                    TextCandidate(
                        item_id=paragraph.paragraph_id,
                        text=f"{paragraph.title}. {paragraph.text}",
                        metadata={
                            "title": paragraph.title,
                            "source_ref": paragraph.title,
                            "sequence_index": paragraph.paragraph_index,
                            "position": paragraph.position,
                        },
                    )
                    for paragraph in record.candidate_paragraphs
                ),
            )
            for record in _MUSIQUE_RANKINGS.validate_python(records)
        ]
    if dataset == "isetrace":
        return [
            TextRankingRequest(
                task_id=record.task_id,
                query_text=record.query_text,
                candidates=(
                    record.provenance_candidates
                    if isetrace_representation == "provenance"
                    else record.flat_candidates
                ),
            )
            for record in _ISETRACE_RANKINGS.validate_python(records)
        ]
    _unsupported_dataset(dataset)


def evidence_graph_build_requests_for_dataset(
    dataset: DatasetId,
    records: Sequence[object],
) -> list[EvidenceGraphBuildRequest]:
    if dataset == "hotpotqa":
        return [
            _sentence_graph_request(record)
            for record in _HOTPOT_RANKINGS.validate_python(records)
        ]
    if dataset == "twowiki":
        return [
            _sentence_graph_request(record)
            for record in _TWOWIKI_RANKINGS.validate_python(records)
        ]
    if dataset == "musique":
        return [
            EvidenceGraphBuildRequest(
                task_id=record.task_id,
                query_text=record.question,
                nodes=tuple(
                    EvidenceGraphBuildNode(
                        node_id=paragraph.paragraph_id,
                        text=paragraph.text,
                        node_kind="document_paragraph",
                        source_ref=paragraph.title,
                        group_key=f"document:{paragraph.title}",
                        sequence_index=paragraph.paragraph_index,
                        metadata={
                            "title": paragraph.title,
                            "position": paragraph.position,
                        },
                    )
                    for paragraph in record.candidate_paragraphs
                ),
                input_visible_edges=(),
            )
            for record in _MUSIQUE_RANKINGS.validate_python(records)
        ]
    if dataset == "isetrace":
        raise TypeError(
            "isetrace uses native provenance views and span evaluation; "
            "EvidenceGraph construction is unsupported"
        )
    _unsupported_dataset(dataset)


def evidence_evaluation_request_for_dataset(
    dataset: DatasetId,
    *,
    predictions: Sequence[RankedResult],
    labels: Sequence[object],
    graphs: Sequence[EvidenceGraph],
) -> EvidenceEvaluationRequest | SpanEvidenceEvaluationRequest:
    if dataset == "isetrace":
        return SpanEvidenceEvaluationRequest(
            predictions=tuple(predictions),
            labels=tuple(
                SpanEvidenceLabel(
                    task_id=label.task_id,
                    gold_evidence_spans=label.gold_evidence_spans,
                )
                for label in _ISETRACE_LABELS.validate_python(labels)
            ),
        )
    return EvidenceEvaluationRequest(
        predictions=tuple(predictions),
        labels=tuple(evidence_labels_for_dataset(dataset, labels)),
        graphs=tuple(graphs),
    )


def evidence_labels_for_dataset(
    dataset: DatasetId,
    labels: Sequence[object],
) -> list[EvidenceLabel]:
    if dataset == "hotpotqa":
        return [
            EvidenceLabel(
                task_id=label.task_id,
                gold_answer=label.gold_answer,
                gold_evidence_item_ids=label.gold_evidence_sentence_ids,
                gold_dependency_edges=label.gold_dependency_edges,
            )
            for label in _HOTPOT_LABELS.validate_python(labels)
        ]
    if dataset == "twowiki":
        return [
            EvidenceLabel(
                task_id=label.task_id,
                gold_answer=label.gold_answer,
                gold_evidence_item_ids=label.gold_evidence_sentence_ids,
                gold_dependency_edges=label.gold_dependency_edges,
            )
            for label in _TWOWIKI_LABELS.validate_python(labels)
        ]
    if dataset == "musique":
        return [
            EvidenceLabel(
                task_id=label.task_id,
                gold_answer=label.gold_answer,
                gold_evidence_item_ids=label.gold_evidence_paragraph_ids,
                gold_dependency_edges=label.gold_dependency_edges,
            )
            for label in _MUSIQUE_LABELS.validate_python(labels)
        ]
    if dataset == "isetrace":
        raise TypeError(
            "isetrace span labels are evaluation-only; training methods are unsupported"
        )
    _unsupported_dataset(dataset)


def _sentence_text_request(
    record: HotpotQARankingRecord | TwoWikiRankingRecord,
) -> TextRankingRequest:
    question_type = (
        {"question_type": record.question_type}
        if isinstance(record, TwoWikiRankingRecord)
        else {}
    )
    return TextRankingRequest(
        task_id=record.task_id,
        query_text=record.question,
        candidates=tuple(
            TextCandidate(
                item_id=sentence.sentence_id,
                text=f"{sentence.title}. {sentence.text}",
                metadata={
                    "title": sentence.title,
                    "source_ref": sentence.title,
                    "sequence_index": sentence.sentence_index,
                    "position": sentence.position,
                    **question_type,
                },
            )
            for sentence in record.candidate_sentences
        ),
    )


def _sentence_graph_request(
    record: HotpotQARankingRecord | TwoWikiRankingRecord,
) -> EvidenceGraphBuildRequest:
    question_type = (
        {"question_type": record.question_type}
        if isinstance(record, TwoWikiRankingRecord)
        else {}
    )
    return EvidenceGraphBuildRequest(
        task_id=record.task_id,
        query_text=record.question,
        nodes=tuple(
            EvidenceGraphBuildNode(
                node_id=sentence.sentence_id,
                text=sentence.text,
                node_kind="document_sentence",
                source_ref=sentence.title,
                group_key=f"document:{sentence.title}",
                sequence_index=sentence.sentence_index,
                metadata={
                    "title": sentence.title,
                    "position": sentence.position,
                    **question_type,
                },
            )
            for sentence in record.candidate_sentences
        ),
        input_visible_edges=(),
    )


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
