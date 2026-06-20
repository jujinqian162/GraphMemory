from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeAlias

from graph_memory.contracts.common import JsonValue, TaskId
from graph_memory.contracts.graphs import MemoryGraph

if TYPE_CHECKING:
    from graph_memory.embeddings.contracts import SentenceEncoder

JsonScalar = str | int | float | bool | None


class DenseConfigLike(Protocol):
    @property
    def model_name(self) -> str:
        ...

    @property
    def query_prefix(self) -> str:
        ...

    @property
    def passage_prefix(self) -> str:
        ...

    @property
    def batch_size(self) -> int:
        ...


@dataclass(frozen=True)
class TextCandidate:
    item_id: str
    text: str
    metadata: Mapping[str, JsonScalar]


@dataclass(frozen=True)
class TextRankingRequest:
    task_id: TaskId
    query_text: str
    candidates: Sequence[TextCandidate]


@dataclass(frozen=True)
class GraphRankingRequest:
    task_id: TaskId
    query_text: str
    candidates: Sequence[TextCandidate]
    graph: MemoryGraph
    initial_scores: Mapping[str, float]


@dataclass(frozen=True)
class TemporalMemoryRankingRequest:
    task_id: TaskId
    query_text: str
    candidates: Sequence[TextCandidate]
    importance_by_item_id: Mapping[str, float]
    metadata: Mapping[str, JsonValue]


RankingMethodRequest: TypeAlias = TextRankingRequest | GraphRankingRequest | TemporalMemoryRankingRequest


@dataclass(frozen=True)
class DenseRuntime:
    config: DenseConfigLike
    encoder: SentenceEncoder | None = None


__all__ = [
    "DenseConfigLike",
    "DenseRuntime",
    "GraphRankingRequest",
    "RankingMethodRequest",
    "TemporalMemoryRankingRequest",
    "TextCandidate",
    "TextRankingRequest",
]