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


class CombinedMuSiQueRecord(DomainModel):
    task_id: NonEmptyStr
    question: NonEmptyStr
    candidate_paragraphs: tuple[MuSiQueCandidateParagraph, ...]
    gold_answer: NonEmptyStr
    gold_answer_aliases: tuple[str, ...]
    gold_evidence_paragraph_ids: tuple[NonEmptyStr, ...]
    gold_dependency_edges: tuple[tuple[NonEmptyStr, NonEmptyStr], ...]
    metadata: dict[str, JsonValue]
    debug: dict[str, JsonValue] | None = None


class MuSiQuePreparedSplit(DomainModel):
    rankings: tuple[MuSiQueRankingRecord, ...]
    labels: tuple[MuSiQueLabelRecord, ...]

    @model_validator(mode="after")
    def _validate_alignment(self) -> "MuSiQuePreparedSplit":
        ranking_by_id = {record.task_id: record for record in self.rankings}
        label_by_id = {record.task_id: record for record in self.labels}
        if len(ranking_by_id) != len(self.rankings):
            raise ValueError("MuSiQue ranking task IDs must be unique")
        if len(label_by_id) != len(self.labels):
            raise ValueError("MuSiQue label task IDs must be unique")
        if set(ranking_by_id) != set(label_by_id):
            raise ValueError("MuSiQue ranking and label task IDs must align")
        for task_id, label in label_by_id.items():
            valid = ranking_by_id[task_id].candidate_ids
            missing = set(label.gold_evidence_paragraph_ids) - valid
            if missing:
                raise ValueError(
                    f"task_id={task_id} gold paragraphs do not exist: {sorted(missing)}"
                )
            for source, target in label.gold_dependency_edges:
                if source not in valid or target not in valid:
                    raise ValueError(
                        f"task_id={task_id} gold dependency edge references "
                        "a missing candidate"
                    )
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


@dataclass(frozen=True)
class ConvertedMuSiQueExample:
    ranking_record: MuSiQueRankingRecord
    label_record: MuSiQueLabelRecord


@dataclass(frozen=True)
class MuSiQueConversionResult:
    ranking_records: list[MuSiQueRankingRecord]
    label_records: list[MuSiQueLabelRecord]


__all__ = [
    "CombinedMuSiQueRecord",
    "ConvertedMuSiQueExample",
    "MuSiQueCandidateParagraph",
    "MuSiQueConversionResult",
    "MuSiQueDecompositionStep",
    "MuSiQueExample",
    "MuSiQueLabelRecord",
    "MuSiQueParagraph",
    "MuSiQuePreparedSplit",
    "MuSiQueRankingRecord",
]
