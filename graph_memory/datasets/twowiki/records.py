from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field, JsonValue, model_validator

from graph_memory.contracts.model import (
    DomainModel,
    NonEmptyStr,
    NonNegativeInt,
    reject_label_fields,
)


class TwoWikiCandidateSentence(DomainModel):
    sentence_id: NonEmptyStr
    title: NonEmptyStr
    sentence_index: NonNegativeInt
    position: NonNegativeInt
    text: NonEmptyStr


class TwoWikiRankingRecord(DomainModel):
    task_id: NonEmptyStr
    question: NonEmptyStr
    question_type: NonEmptyStr
    candidate_sentences: tuple[TwoWikiCandidateSentence, ...] = Field(min_length=1)
    metadata: dict[str, JsonValue]
    debug: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def _validate_ranking(self) -> "TwoWikiRankingRecord":
        if not self.task_id.startswith("2wiki_"):
            raise ValueError("2Wiki task_id must start with 2wiki_")
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
            path="2Wiki ranking record",
        )
        return self

    @property
    def candidate_ids(self) -> frozenset[str]:
        return frozenset(item.sentence_id for item in self.candidate_sentences)


class TwoWikiLabelRecord(DomainModel):
    task_id: NonEmptyStr
    gold_answer: NonEmptyStr
    gold_evidence_sentence_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    gold_dependency_edges: tuple[tuple[NonEmptyStr, NonEmptyStr], ...]
    metadata: dict[str, JsonValue]
    debug: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def _validate_label(self) -> "TwoWikiLabelRecord":
        if len(self.gold_evidence_sentence_ids) != len(
            set(self.gold_evidence_sentence_ids)
        ):
            raise ValueError("gold evidence sentence IDs must be unique")
        return self


@dataclass(frozen=True)
class TwoWikiDocument:
    title: str
    sentences: tuple[str, ...]


@dataclass(frozen=True)
class TwoWikiSupportingFact:
    title: str
    sentence_id: int


@dataclass(frozen=True)
class TwoWikiEvidenceTriple:
    subject: str
    relation: str
    object: str


@dataclass(frozen=True)
class TwoWikiExample:
    raw_id: str
    question: str
    answer: str
    question_type: str
    documents: tuple[TwoWikiDocument, ...]
    supporting_facts: tuple[TwoWikiSupportingFact, ...]
    evidences: tuple[TwoWikiEvidenceTriple, ...]
    evidences_id: tuple[TwoWikiEvidenceTriple, ...]
    answer_id: str | None


__all__ = [
    "TwoWikiCandidateSentence",
    "TwoWikiDocument",
    "TwoWikiEvidenceTriple",
    "TwoWikiExample",
    "TwoWikiLabelRecord",
    "TwoWikiRankingRecord",
    "TwoWikiSupportingFact",
]
