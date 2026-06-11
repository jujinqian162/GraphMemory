from __future__ import annotations

from dataclasses import dataclass

from graph_memory.contracts.common import NodeId, TaskId, TrainPairSampleType


@dataclass(frozen=True)
class DenseFinetuneDataSettings:
    hard_negatives_per_positive: int = 1

    def __post_init__(self) -> None:
        if self.hard_negatives_per_positive < 0:
            raise ValueError("hard_negatives_per_positive must be non-negative.")


@dataclass(frozen=True)
class DenseFinetuneExample:
    task_id: TaskId
    positive_node_id: NodeId
    negative_node_id: NodeId | None
    anchor: str
    positive: str
    negative: str | None
    negative_sample_type: TrainPairSampleType | None


@dataclass(frozen=True)
class DenseFinetuneDatasetBuildResult:
    examples: tuple[DenseFinetuneExample, ...]
    rows: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class DenseFinetuneIREvaluatorPayload:
    queries: dict[str, str]
    corpus: dict[str, str]
    relevant_docs: dict[str, set[str]]
