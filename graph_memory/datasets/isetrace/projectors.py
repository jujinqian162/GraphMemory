from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from graph_memory.datasets.isetrace.benchmark_records import (
    ISETraceLabelRecord,
    ISETraceRankingRecord,
)
from graph_memory.evaluation.requests import (
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
                    gold_evidence_spans=label.gold_evidence_spans,
                )
                for label in labels
            ),
        )


__all__ = [
    "ISETraceToSpanEvidenceEvaluationRequest",
    "ISETraceToTextRankingRequest",
]
