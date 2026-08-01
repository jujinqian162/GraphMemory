from __future__ import annotations

from pydantic import Field, model_validator

from graph_memory.contracts.model import DomainModel, NonEmptyStr
from graph_memory.graphs.contracts import EvidenceGraph
from graph_memory.retrieval.results import RankedResult
from graph_memory.trajectories import SourceSpan


class EvidenceLabel(DomainModel):
    task_id: NonEmptyStr
    gold_answer: str
    gold_evidence_item_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    gold_dependency_edges: tuple[tuple[NonEmptyStr, NonEmptyStr], ...]
    query_intent: NonEmptyStr | None = None
    motif_type: NonEmptyStr | None = None
    review_status: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _validate_label(self) -> "EvidenceLabel":
        gold = set(self.gold_evidence_item_ids)
        if len(gold) != len(self.gold_evidence_item_ids):
            raise ValueError("gold evidence item IDs must be unique")
        for source, target in self.gold_dependency_edges:
            if source not in gold or target not in gold:
                raise ValueError(
                    "gold dependency edges must reference gold evidence items"
                )
        return self


class SpanEvidenceLabel(DomainModel):
    task_id: NonEmptyStr
    gold_evidence_spans: tuple[SourceSpan, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_spans(self) -> "SpanEvidenceLabel":
        keys = []
        for span in self.gold_evidence_spans:
            if span.char_start is None or span.char_end is None:
                raise ValueError("gold evidence requires exact character spans")
            if span.json_pointer is None:
                raise ValueError("gold evidence requires a source json_pointer")
            keys.append(
                (span.event_id, span.json_pointer, span.char_start, span.char_end)
            )
        if len(keys) != len(set(keys)):
            raise ValueError("gold evidence spans must be unique")
        return self


class SpanEvidenceEvaluationRequest(DomainModel):
    predictions: tuple[RankedResult, ...]
    labels: tuple[SpanEvidenceLabel, ...]

    @model_validator(mode="after")
    def _validate_alignment(self) -> "SpanEvidenceEvaluationRequest":
        label_ids = [label.task_id for label in self.labels]
        prediction_ids = [prediction.task_id for prediction in self.predictions]
        if len(label_ids) != len(set(label_ids)):
            raise ValueError("span evaluation label task IDs must be unique")
        if len(prediction_ids) != len(set(prediction_ids)):
            raise ValueError("span evaluation prediction task IDs must be unique")
        if set(label_ids) != set(prediction_ids):
            raise ValueError("span evaluation predictions and labels must align")
        for prediction in self.predictions:
            for ranked_node in prediction.ranked_nodes:
                if not ranked_node.source_spans:
                    raise ValueError(
                        f"task_id={prediction.task_id} ranked node="
                        f"{ranked_node.node_id} lacks source spans"
                    )
                if any(
                    span.char_start is None
                    or span.char_end is None
                    or span.json_pointer is None
                    for span in ranked_node.source_spans
                ):
                    raise ValueError(
                        f"task_id={prediction.task_id} ranked node="
                        f"{ranked_node.node_id} has non-exact source spans"
                    )
        return self


class EvidenceEvaluationRequest(DomainModel):
    predictions: tuple[RankedResult, ...]
    labels: tuple[EvidenceLabel, ...]
    graphs: tuple[EvidenceGraph, ...]

    @model_validator(mode="after")
    def _validate_alignment(self) -> "EvidenceEvaluationRequest":
        labels_by_id = {label.task_id: label for label in self.labels}
        if len(labels_by_id) != len(self.labels):
            raise ValueError("evaluation label task IDs must be unique")
        expected = set(labels_by_id)
        if self.predictions:
            prediction_ids = [prediction.task_id for prediction in self.predictions]
            if len(prediction_ids) != len(set(prediction_ids)):
                raise ValueError("evaluation prediction task IDs must be unique")
            if set(prediction_ids) != expected:
                raise ValueError("evaluation predictions and labels must align")
        if self.graphs:
            graph_ids = [graph.task_id for graph in self.graphs]
            if len(graph_ids) != len(set(graph_ids)):
                raise ValueError("evaluation graph task IDs must be unique")
            if set(graph_ids) != expected:
                raise ValueError("evaluation graphs and labels must align")
            for graph in self.graphs:
                missing = (
                    set(labels_by_id[graph.task_id].gold_evidence_item_ids)
                    - graph.graph_item_ids
                )
                if missing:
                    raise ValueError(
                        f"task_id={graph.task_id} gold evidence missing from graph: "
                        f"{sorted(missing)}"
                    )
        return self


__all__ = [
    "EvidenceEvaluationRequest",
    "EvidenceLabel",
    "SpanEvidenceEvaluationRequest",
    "SpanEvidenceLabel",
]
