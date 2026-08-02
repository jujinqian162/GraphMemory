from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from graph_memory.graphs.provenance import ProvenanceGraph
from graph_memory.query_synthesis.provenance.contracts import TemplateSupervisionRecord
from graph_memory.retrieval.requests import TextCandidate
from graph_memory.trajectories import SourceSpan
from graph_memory.contracts.model import DomainModel, NonEmptyStr


class ISETraceRankingRecord(DomainModel):
    task_id: NonEmptyStr
    graph_id: NonEmptyStr
    query_text: NonEmptyStr
    flat_candidates: tuple[TextCandidate, ...] = Field(min_length=1)
    provenance_candidates: tuple[TextCandidate, ...] = Field(min_length=1)

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


class ISETraceQueryMetadata(DomainModel):
    task_id: NonEmptyStr
    graph_id: NonEmptyStr
    query_origin: Literal["natural", "template"]


class ISETraceLabelRecord(DomainModel):
    task_id: NonEmptyStr
    graph_id: NonEmptyStr
    gold_evidence_spans: tuple[SourceSpan, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_label(self) -> "ISETraceLabelRecord":
        keys = [
            (span.event_id, span.json_pointer, span.char_start, span.char_end)
            for span in self.gold_evidence_spans
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("gold evidence spans must be unique")
        if any(
            span.char_start is None
            or span.char_end is None
            or span.json_pointer is None
            for span in self.gold_evidence_spans
        ):
            raise ValueError("gold evidence spans must be exact source spans")
        return self


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
    query_metadata: tuple[ISETraceQueryMetadata, ...]
    template_supervision: tuple[TemplateSupervisionRecord, ...]
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
        metadata_ids = [record.task_id for record in self.query_metadata]
        if len(metadata_ids) != len(set(metadata_ids)):
            raise ValueError("ISETrace query metadata task IDs must be unique")
        if set(metadata_ids) != set(task_ids):
            raise ValueError("ISETrace ranking and query metadata tasks must align")
        template_ids = [record.task_id for record in self.template_supervision]
        if len(template_ids) != len(set(template_ids)):
            raise ValueError("ISETrace template supervision task IDs must be unique")
        if set(template_ids) - set(task_ids):
            raise ValueError("ISETrace template supervision has unknown task IDs")
        origins = {record.task_id: record.query_origin for record in self.query_metadata}
        if any(origins[task_id] != "template" for task_id in template_ids):
            raise ValueError("ISETrace template supervision must have template origin")
        graph_by_id = {graph.graph_id: graph for graph in self.provenance_graphs}
        if len(graph_by_id) != len(self.provenance_graphs):
            raise ValueError("ISETrace provenance graph IDs must be unique")
        expected_graphs = {record.graph_id for record in self.rankings}
        if set(graph_by_id) != expected_graphs:
            raise ValueError("ISETrace task and provenance graph coverage must align")
        return self


__all__ = [
    "CombinedISETraceBenchmarkRecord",
    "ISETraceLabelRecord",
    "ISETracePreparedBenchmark",
    "ISETraceQueryMetadata",
    "ISETraceRankingRecord",
]
