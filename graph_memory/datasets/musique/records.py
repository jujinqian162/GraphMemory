from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field, JsonValue, model_validator

from graph_memory.contracts.model import (
    DomainModel,
    NonEmptyStr,
    NonNegativeInt,
    reject_label_fields,
)


class MuSiQueCandidateParagraph(DomainModel):
    paragraph_id: NonEmptyStr
    title: NonEmptyStr
    paragraph_index: NonNegativeInt
    position: NonNegativeInt
    text: NonEmptyStr


class MuSiQueRankingRecord(DomainModel):
    task_id: NonEmptyStr
    question: NonEmptyStr
    candidate_paragraphs: tuple[MuSiQueCandidateParagraph, ...] = Field(min_length=1)
    metadata: dict[str, JsonValue]
    debug: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def _validate_ranking(self) -> "MuSiQueRankingRecord":
        if not self.task_id.startswith("musique_"):
            raise ValueError("MuSiQue task_id must start with musique_")
        candidate_ids: list[str] = []
        for expected, candidate in enumerate(self.candidate_paragraphs):
            candidate_ids.append(candidate.paragraph_id)
            if candidate.position != expected:
                raise ValueError(
                    f"paragraph_id={candidate.paragraph_id} position="
                    f"{candidate.position} expected {expected}"
                )
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("MuSiQue paragraph IDs must be unique")
        reject_label_fields(
            self.model_dump(mode="python", exclude_none=True),
            path="MuSiQue ranking record",
        )
        return self

    @property
    def candidate_ids(self) -> frozenset[str]:
        return frozenset(item.paragraph_id for item in self.candidate_paragraphs)


class MuSiQueLabelRecord(DomainModel):
    task_id: NonEmptyStr
    gold_answer: NonEmptyStr
    gold_answer_aliases: tuple[str, ...]
    gold_evidence_paragraph_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    gold_dependency_edges: tuple[tuple[NonEmptyStr, NonEmptyStr], ...]
    metadata: dict[str, JsonValue]
    debug: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def _validate_label(self) -> "MuSiQueLabelRecord":
        if len(self.gold_evidence_paragraph_ids) != len(
            set(self.gold_evidence_paragraph_ids)
        ):
            raise ValueError("gold evidence paragraph IDs must be unique")
        return self


@dataclass(frozen=True)
class MuSiQueParagraph:
    idx: int
    title: str
    paragraph_text: str
    is_supporting: bool


@dataclass(frozen=True)
class MuSiQueDecompositionStep:
    step_id: str
    question: str
    answer: str
    paragraph_support_idx: int | None


@dataclass(frozen=True)
class MuSiQueExample:
    raw_id: str
    question: str
    answer: str
    answer_aliases: tuple[str, ...]
    answerable: bool
    paragraphs: tuple[MuSiQueParagraph, ...]
    question_decomposition: tuple[MuSiQueDecompositionStep, ...]


__all__ = [
    "MuSiQueCandidateParagraph",
    "MuSiQueDecompositionStep",
    "MuSiQueExample",
    "MuSiQueLabelRecord",
    "MuSiQueParagraph",
    "MuSiQueRankingRecord",
]
