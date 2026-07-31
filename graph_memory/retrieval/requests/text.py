from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from pydantic import model_validator

from graph_memory.contracts.model import DomainModel, NonEmptyStr
from graph_memory.trajectories import SourceSpan

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


class TextCandidate(DomainModel):
    item_id: NonEmptyStr
    text: str
    metadata: dict[str, JsonScalar]
    source_spans: tuple[SourceSpan, ...] = ()


class TextRankingRequest(DomainModel):
    task_id: NonEmptyStr
    query_text: str
    candidates: tuple[TextCandidate, ...]

    @model_validator(mode="after")
    def _unique_candidates(self) -> "TextRankingRequest":
        candidate_ids = [candidate.item_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("text ranking candidate IDs must be unique")
        return self

    @property
    def candidate_ids(self) -> frozenset[str]:
        return frozenset(candidate.item_id for candidate in self.candidates)


@dataclass(frozen=True)
class DenseRuntime:
    """Opaque runtime dependency bundle, not a persisted artifact contract."""

    config: DenseConfigLike
    encoder: SentenceEncoder | None = None


__all__ = [
    "DenseConfigLike",
    "DenseRuntime",
    "JsonScalar",
    "TextCandidate",
    "TextRankingRequest",
]
