from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from graph_memory.contracts.common import TaskId

if TYPE_CHECKING:
    from graph_memory.embeddings.contracts import SentenceEncoder

JsonScalar = str | int | float | bool | None


class DenseConfigLike(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def query_prefix(self) -> str: ...

    @property
    def passage_prefix(self) -> str: ...

    @property
    def batch_size(self) -> int: ...


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
class DenseRuntime:
    config: DenseConfigLike
    encoder: SentenceEncoder | None = None


__all__ = [
    "DenseConfigLike",
    "DenseRuntime",
    "JsonScalar",
    "TextCandidate",
    "TextRankingRequest",
]
