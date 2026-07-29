from __future__ import annotations

from pydantic import Field, model_validator

from graph_memory.contracts.model import DomainModel, NonEmptyStr
from graph_memory.graphs.contracts import EvidenceGraph
from graph_memory.retrieval.results import RankedResult


class EvidenceLabel(DomainModel):
    task_id: NonEmptyStr
    gold_answer: str
    gold_evidence_item_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    gold_dependency_edges: tuple[tuple[NonEmptyStr, NonEmptyStr], ...]

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
                missing = set(
                    labels_by_id[graph.task_id].gold_evidence_item_ids
                ) - graph.graph_item_ids
                if missing:
                    raise ValueError(
                        f"task_id={graph.task_id} gold evidence missing from graph: "
                        f"{sorted(missing)}"
                    )
        return self


__all__ = ["EvidenceEvaluationRequest", "EvidenceLabel"]
