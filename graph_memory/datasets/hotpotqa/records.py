from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field, JsonValue, model_validator

from graph_memory.contracts.model import (
    DomainModel,
    NonEmptyStr,
    NonNegativeInt,
    reject_label_fields,
)


class HotpotQACandidateSentence(DomainModel):
    sentence_id: NonEmptyStr
    title: NonEmptyStr
    sentence_index: NonNegativeInt
    position: NonNegativeInt
    text: NonEmptyStr


class HotpotQARankingRecord(DomainModel):
    task_id: NonEmptyStr
    question: NonEmptyStr
    candidate_sentences: tuple[HotpotQACandidateSentence, ...] = Field(min_length=1)
    metadata: dict[str, JsonValue] | None = None
    debug: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def _validate_candidates(self) -> "HotpotQARankingRecord":
        for expected, candidate in enumerate(self.candidate_sentences):
            if candidate.sentence_id != f"m{expected}":
                raise ValueError(
                    f"sentence_id={candidate.sentence_id} expected m{expected}"
                )
            if candidate.position != expected:
                raise ValueError(
                    f"sentence_id={candidate.sentence_id} position="
                    f"{candidate.position} expected {expected}"
                )
        reject_label_fields(
            self.model_dump(mode="python", exclude_none=True),
            path="HotpotQA ranking record",
        )
        return self

    @property
    def candidate_ids(self) -> frozenset[str]:
        return frozenset(item.sentence_id for item in self.candidate_sentences)


class HotpotQALabelRecord(DomainModel):
    task_id: NonEmptyStr
    gold_answer: NonEmptyStr
    gold_evidence_sentence_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    gold_dependency_edges: tuple[tuple[NonEmptyStr, NonEmptyStr], ...]
    metadata: dict[str, JsonValue] | None = None
    debug: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def _validate_label(self) -> "HotpotQALabelRecord":
        if len(self.gold_evidence_sentence_ids) != len(
            set(self.gold_evidence_sentence_ids)
        ):
            raise ValueError("gold evidence sentence IDs must be unique")
        if self.gold_dependency_edges:
            raise ValueError("HotpotQA gold dependency edges must be empty")
        return self


class CombinedHotpotQARecord(DomainModel):
    task_id: NonEmptyStr
    question: NonEmptyStr
    candidate_sentences: tuple[HotpotQACandidateSentence, ...]
    gold_answer: NonEmptyStr
    gold_evidence_sentence_ids: tuple[NonEmptyStr, ...]
    gold_dependency_edges: tuple[tuple[NonEmptyStr, NonEmptyStr], ...]
    metadata: dict[str, JsonValue] | None = None
    debug: dict[str, JsonValue] | None = None


class HotpotQAPreparedSplit(DomainModel):
    rankings: tuple[HotpotQARankingRecord, ...]
    labels: tuple[HotpotQALabelRecord, ...]

    @model_validator(mode="after")
    def _validate_alignment(self) -> "HotpotQAPreparedSplit":
        _validate_split_alignment(self.rankings, self.labels)
        return self


@dataclass(frozen=True)
class HotpotQADocument:
    title: str
    sentences: tuple[str, ...]


@dataclass(frozen=True)
class HotpotQASupportingFact:
    title: str
    sentence_id: int


@dataclass(frozen=True)
class HotpotQAExample:
    raw_id: str
    question: str
    answer: str
    documents: tuple[HotpotQADocument, ...]
    supporting_facts: tuple[HotpotQASupportingFact, ...]


@dataclass(frozen=True)
class ConvertedHotpotQAExample:
    ranking_record: HotpotQARankingRecord
    label_record: HotpotQALabelRecord


@dataclass(frozen=True)
class HotpotQAConversionResult:
    ranking_records: list[HotpotQARankingRecord]
    label_records: list[HotpotQALabelRecord]


def _validate_split_alignment(
    rankings: tuple[HotpotQARankingRecord, ...],
    labels: tuple[HotpotQALabelRecord, ...],
) -> None:
    ranking_by_id = {record.task_id: record for record in rankings}
    label_by_id = {record.task_id: record for record in labels}
    if len(ranking_by_id) != len(rankings):
        raise ValueError("HotpotQA ranking task IDs must be unique")
    if len(label_by_id) != len(labels):
        raise ValueError("HotpotQA label task IDs must be unique")
    if set(ranking_by_id) != set(label_by_id):
        raise ValueError("HotpotQA ranking and label task IDs must align")
    for task_id, label in label_by_id.items():
        missing = set(label.gold_evidence_sentence_ids) - ranking_by_id[
            task_id
        ].candidate_ids
        if missing:
            raise ValueError(
                f"task_id={task_id} gold sentences do not exist: {sorted(missing)}"
            )


__all__ = [
    "CombinedHotpotQARecord",
    "ConvertedHotpotQAExample",
    "HotpotQACandidateSentence",
    "HotpotQAConversionResult",
    "HotpotQADocument",
    "HotpotQAExample",
    "HotpotQALabelRecord",
    "HotpotQAPreparedSplit",
    "HotpotQARankingRecord",
    "HotpotQASupportingFact",
]
