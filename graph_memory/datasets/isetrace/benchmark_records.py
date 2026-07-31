from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import Field, model_validator

from graph_memory.contracts.model import DomainModel, NonEmptyStr
from graph_memory.graphs.provenance import OutputDependency, ProvenanceGraph
from graph_memory.query_synthesis.provenance.contracts import MotifType, QueryIntent
from graph_memory.retrieval.requests import TextCandidate
from graph_memory.trajectories import SourceSpan

ISETraceReviewPolicy: TypeAlias = Literal["allow_unreviewed", "accepted_only"]
ISETraceLabelPolicy: TypeAlias = Literal["answer_only", "support", "intent_aware"]


class ISETraceRankingRecord(DomainModel):
    task_id: NonEmptyStr
    graph_id: NonEmptyStr
    query_text: NonEmptyStr
    flat_candidates: tuple[TextCandidate, ...] = Field(min_length=1)
    provenance_candidates: tuple[TextCandidate, ...] = Field(min_length=1)
    logical_dependencies: tuple[OutputDependency, ...]

    @model_validator(mode="after")
    def _validate_ranking(self) -> "ISETraceRankingRecord":
        for name, candidates in (
            ("flat_candidates", self.flat_candidates),
            ("provenance_candidates", self.provenance_candidates),
        ):
            candidate_ids = [candidate.item_id for candidate in candidates]
            if len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError(f"ISETrace {name} IDs must be unique")
            for candidate in candidates:
                if not candidate.source_spans:
                    raise ValueError(f"ISETrace {name} require source spans")
        return self


class ISETraceLabelRecord(DomainModel):
    task_id: NonEmptyStr
    graph_id: NonEmptyStr
    gold_answer: str
    gold_evidence_output_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    gold_evidence_spans: tuple[SourceSpan, ...] = Field(min_length=1)
    gold_evidence_spans_by_output_id: dict[
        NonEmptyStr, tuple[SourceSpan, ...]
    ]
    gold_dependency_edges: tuple[tuple[NonEmptyStr, NonEmptyStr], ...]
    answer_output_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    answer_evidence_spans: tuple[SourceSpan, ...] = Field(min_length=1)
    support_output_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    support_evidence_spans: tuple[SourceSpan, ...] = Field(min_length=1)
    motif_id: NonEmptyStr
    motif_type: MotifType
    query_intent: QueryIntent
    authoring_review_status: Literal[
        "template", "unreviewed", "accepted", "edited", "rejected"
    ]
    label_policy: ISETraceLabelPolicy

    @model_validator(mode="after")
    def _validate_label(self) -> "ISETraceLabelRecord":
        for name, values in (
            ("gold_evidence_output_ids", self.gold_evidence_output_ids),
            ("answer_output_ids", self.answer_output_ids),
            ("support_output_ids", self.support_output_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")
        for name, spans in (
            ("gold_evidence_spans", self.gold_evidence_spans),
            ("answer_evidence_spans", self.answer_evidence_spans),
            ("support_evidence_spans", self.support_evidence_spans),
        ):
            keys = [
                (span.event_id, span.json_pointer, span.char_start, span.char_end)
                for span in spans
            ]
            if len(keys) != len(set(keys)):
                raise ValueError(f"{name} must be unique")
            if any(span.char_start is None or span.char_end is None for span in spans):
                raise ValueError(f"{name} must contain exact character spans")
        gold = set(self.gold_evidence_output_ids)
        if set(self.gold_evidence_spans_by_output_id) != gold:
            raise ValueError(
                "gold span mapping must cover every gold output exactly"
            )
        if any(not spans for spans in self.gold_evidence_spans_by_output_id.values()):
            raise ValueError("gold span mapping values must be non-empty")
        mapped_span_keys = {
            key
            for spans in self.gold_evidence_spans_by_output_id.values()
            for key in _span_keys(spans)
        }
        if mapped_span_keys != set(_span_keys(self.gold_evidence_spans)):
            raise ValueError("gold span mapping must equal gold evidence spans")
        if not set(self.answer_output_ids).issubset(self.support_output_ids):
            raise ValueError("answer outputs must be included in support outputs")
        if not set(_span_keys(self.answer_evidence_spans)).issubset(
            _span_keys(self.support_evidence_spans)
        ):
            raise ValueError("answer evidence spans must be included in support spans")
        if not set(_span_keys(self.gold_evidence_spans)).issubset(
            _span_keys(self.support_evidence_spans)
        ):
            raise ValueError("gold evidence spans must be included in support spans")
        for source, target in self.gold_dependency_edges:
            if source not in gold or target not in gold:
                raise ValueError("gold dependency edges must stay inside gold evidence")
        return self


def _span_keys(spans: tuple[SourceSpan, ...]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (span.event_id, span.json_pointer, span.char_start, span.char_end)
        for span in spans
    )


class CombinedISETraceBenchmarkRecord(DomainModel):
    ranking: ISETraceRankingRecord
    label: ISETraceLabelRecord

    @model_validator(mode="after")
    def _validate_alignment(self) -> "CombinedISETraceBenchmarkRecord":
        if self.ranking.task_id != self.label.task_id:
            raise ValueError("combined ISETrace task IDs must align")
        if self.ranking.graph_id != self.label.graph_id:
            raise ValueError("combined ISETrace graph IDs must align")
        return self


class ISETracePreparedBenchmark(DomainModel):
    rankings: tuple[ISETraceRankingRecord, ...]
    labels: tuple[ISETraceLabelRecord, ...]
    provenance_graphs: tuple[ProvenanceGraph, ...]

    @model_validator(mode="after")
    def _validate_coverage(self) -> "ISETracePreparedBenchmark":
        task_ids = [record.task_id for record in self.rankings]
        label_ids = [record.task_id for record in self.labels]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("ISETrace ranking task IDs must be unique")
        if len(label_ids) != len(set(label_ids)):
            raise ValueError("ISETrace label task IDs must be unique")
        if set(task_ids) != set(label_ids):
            raise ValueError("ISETrace ranking and label tasks must align")
        graph_by_id = {graph.graph_id: graph for graph in self.provenance_graphs}
        if len(graph_by_id) != len(self.provenance_graphs):
            raise ValueError("ISETrace provenance graph IDs must be unique")
        expected_graphs = {record.graph_id for record in self.rankings}
        if set(graph_by_id) != expected_graphs:
            raise ValueError("ISETrace task and provenance graph coverage must align")
        return self


__all__ = [
    "CombinedISETraceBenchmarkRecord",
    "ISETraceLabelPolicy",
    "ISETraceLabelRecord",
    "ISETracePreparedBenchmark",
    "ISETraceRankingRecord",
    "ISETraceReviewPolicy",
]
