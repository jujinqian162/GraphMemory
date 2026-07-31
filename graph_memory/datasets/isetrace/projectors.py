from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from graph_memory.datasets.isetrace.benchmark_records import (
    ISETraceLabelRecord,
    ISETraceRankingRecord,
)
from graph_memory.evaluation.requests import (
    SpanEvidenceDependency,
    SpanEvidenceEvaluationRequest,
    SpanEvidenceLabel,
)
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.retrieval.results import RankedResult


class ISETraceToTextRankingRequest:
    def project(
        self,
        record: ISETraceRankingRecord,
        *,
        representation: Literal["flat", "provenance"],
    ) -> TextRankingRequest:
        candidates = (
            record.flat_candidates
            if representation == "flat"
            else record.provenance_candidates
        )
        return TextRankingRequest(
            task_id=record.task_id,
            query_text=record.query_text,
            candidates=candidates,
        )


class ISETraceToSpanEvidenceEvaluationRequest:
    def project(
        self,
        *,
        predictions: Sequence[RankedResult],
        labels: Sequence[ISETraceLabelRecord],
    ) -> SpanEvidenceEvaluationRequest:
        return SpanEvidenceEvaluationRequest(
            predictions=tuple(predictions),
            labels=tuple(
                SpanEvidenceLabel(
                    task_id=label.task_id,
                    gold_answer=label.gold_answer,
                    gold_evidence_spans=label.gold_evidence_spans,
                    gold_dependency_edges=tuple(
                        SpanEvidenceDependency(
                            source_spans=label.gold_evidence_spans_by_output_id[source],
                            target_spans=label.gold_evidence_spans_by_output_id[target],
                        )
                        for source, target in label.gold_dependency_edges
                    ),
                    query_intent=label.query_intent,
                    motif_type=label.motif_type,
                    review_status=label.authoring_review_status,
                )
                for label in labels
            ),
        )


__all__ = [
    "ISETraceToSpanEvidenceEvaluationRequest",
    "ISETraceToTextRankingRequest",
]
