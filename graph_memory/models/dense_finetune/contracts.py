from __future__ import annotations

from graph_memory.contracts.common import NodeId, TaskId, TrainPairSampleType
from graph_memory.contracts.model import DomainModel, NonNegativeInt


class DenseFinetuneDataSettings(DomainModel):
    hard_negatives_per_positive: NonNegativeInt = 1


class DenseFinetuneExample(DomainModel):
    task_id: TaskId
    positive_node_id: NodeId
    negative_node_id: NodeId | None
    anchor: str
    positive: str
    negative: str | None
    negative_sample_type: TrainPairSampleType | None


class DenseFinetuneDatasetBuildResult(DomainModel):
    examples: tuple[DenseFinetuneExample, ...]
    rows: tuple[dict[str, str], ...]


class DenseFinetuneIREvaluatorPayload(DomainModel):
    queries: dict[str, str]
    corpus: dict[str, str]
    relevant_docs: dict[str, frozenset[str]]
