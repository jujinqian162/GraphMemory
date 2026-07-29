from __future__ import annotations

from typing import Literal, get_args

from pydantic import model_validator

from graph_memory.contracts.common import TrainPairSampleType
from graph_memory.contracts.model import (
    DomainModel,
    NonEmptyStr,
    NonNegativeFiniteFloat,
    NonNegativeInt,
)
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.graphs.contracts import EvidenceGraph
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.training_pairs.config import NegativeSamplingConfig

_NEGATIVE_SAMPLE_TYPES = frozenset(get_args(TrainPairSampleType)) - {"positive"}


class TrainPairRecord(DomainModel):
    task_id: NonEmptyStr
    node_id: NonEmptyStr
    label: Literal[0, 1]
    sample_type: TrainPairSampleType

    @model_validator(mode="after")
    def _validate_semantics(self) -> "TrainPairRecord":
        if self.node_id == "q":
            raise ValueError("query node cannot be a train pair")
        if self.sample_type == "positive" and self.label != 1:
            raise ValueError("positive sample requires label=1")
        if self.sample_type != "positive" and self.label != 0:
            raise ValueError("negative sample requires label=0")
        return self


class TrainPairBuildSummary(DomainModel):
    positive_count: NonNegativeInt
    negative_count_by_type: dict[NonEmptyStr, NonNegativeInt]
    avg_positive_per_task: NonNegativeFiniteFloat
    avg_negative_per_task: NonNegativeFiniteFloat
    tasks_with_no_positive: tuple[NonEmptyStr, ...]
    sampling_config: NegativeSamplingConfig
    requested_negative_count_by_type: dict[NonEmptyStr, NonNegativeInt] | None = None
    shortfall_by_type: dict[NonEmptyStr, NonNegativeInt] | None = None
    overlap_count_by_type: dict[NonEmptyStr, NonNegativeInt] | None = None
    source_overlap_by_task: dict[
        NonEmptyStr, dict[NonEmptyStr, tuple[NonEmptyStr, ...]]
    ] | None = None

    @model_validator(mode="after")
    def _validate_summary(self) -> "TrainPairBuildSummary":
        if self.tasks_with_no_positive:
            raise ValueError("tasks_with_no_positive must be empty")
        unknown = sorted(set(self.negative_count_by_type) - _NEGATIVE_SAMPLE_TYPES)
        if unknown:
            raise ValueError(f"unsupported negative sample types={unknown}")
        for field_name in (
            "requested_negative_count_by_type",
            "shortfall_by_type",
            "overlap_count_by_type",
        ):
            values = getattr(self, field_name)
            if values is not None:
                unknown = sorted(set(values) - _NEGATIVE_SAMPLE_TYPES)
                if unknown:
                    raise ValueError(
                        f"{field_name} has unsupported sample types={unknown}"
                    )
        return self


class TrainPairDataset(DomainModel):
    requests: tuple[TextRankingRequest, ...]
    labels: tuple[EvidenceLabel, ...]
    pairs: tuple[TrainPairRecord, ...]
    graphs: tuple[EvidenceGraph, ...] = ()

    @model_validator(mode="after")
    def _validate_dataset(self) -> "TrainPairDataset":
        requests = {request.task_id: request for request in self.requests}
        labels = {label.task_id: label for label in self.labels}
        if len(requests) != len(self.requests):
            raise ValueError("train-pair request task IDs must be unique")
        if len(labels) != len(self.labels):
            raise ValueError("train-pair label task IDs must be unique")
        if set(requests) != set(labels):
            raise ValueError("train-pair requests and labels must align")
        graphs = {graph.task_id: graph for graph in self.graphs}
        if len(graphs) != len(self.graphs):
            raise ValueError("train-pair graph task IDs must be unique")
        if graphs and set(graphs) != set(requests):
            raise ValueError("train-pair requests and graphs must align")

        seen: set[tuple[str, str, str]] = set()
        positives: dict[str, set[str]] = {task_id: set() for task_id in labels}
        for pair in self.pairs:
            request = requests.get(pair.task_id)
            label = labels.get(pair.task_id)
            if request is None or label is None:
                raise ValueError(f"pair task_id={pair.task_id} has no task context")
            if pair.node_id not in request.candidate_ids:
                raise ValueError(
                    f"task_id={pair.task_id} node_id={pair.node_id} "
                    "does not exist in candidates"
                )
            if graphs and pair.node_id not in graphs[pair.task_id].graph_item_ids:
                raise ValueError(
                    f"task_id={pair.task_id} node_id={pair.node_id} "
                    "does not exist in graph"
                )
            key = (pair.task_id, pair.node_id, pair.sample_type)
            if key in seen:
                raise ValueError(f"duplicate train pair={key}")
            seen.add(key)
            gold = set(label.gold_evidence_item_ids)
            if pair.label == 1:
                if pair.node_id not in gold:
                    raise ValueError(
                        f"task_id={pair.task_id} positive node is not gold evidence"
                    )
                positives[pair.task_id].add(pair.node_id)
            elif pair.node_id in gold:
                raise ValueError(
                    f"task_id={pair.task_id} negative node is gold evidence"
                )
        for task_id, label in labels.items():
            gold = set(label.gold_evidence_item_ids)
            if positives[task_id] != gold:
                missing = sorted(gold - positives[task_id])
                extra = sorted(positives[task_id] - gold)
                raise ValueError(
                    f"task_id={task_id} positives must exactly match gold; "
                    f"missing={missing} extra={extra}"
                )
        return self


class TrainPairBuildResult(DomainModel):
    pairs: tuple[TrainPairRecord, ...]
    summary: TrainPairBuildSummary


__all__ = [
    "TrainPairBuildResult",
    "TrainPairBuildSummary",
    "TrainPairDataset",
    "TrainPairRecord",
]
