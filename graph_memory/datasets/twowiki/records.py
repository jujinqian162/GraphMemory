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


class CombinedTwoWikiRecord(DomainModel):
    task_id: NonEmptyStr
    question: NonEmptyStr
    question_type: NonEmptyStr
    candidate_sentences: tuple[TwoWikiCandidateSentence, ...]
    gold_answer: NonEmptyStr
    gold_evidence_sentence_ids: tuple[NonEmptyStr, ...]
    gold_dependency_edges: tuple[tuple[NonEmptyStr, NonEmptyStr], ...]
    metadata: dict[str, JsonValue]
    debug: dict[str, JsonValue] | None = None


class TwoWikiPreparedSplit(DomainModel):
    rankings: tuple[TwoWikiRankingRecord, ...]
    labels: tuple[TwoWikiLabelRecord, ...]

    @model_validator(mode="after")
    def _validate_alignment(self) -> "TwoWikiPreparedSplit":
        ranking_by_id = {record.task_id: record for record in self.rankings}
        label_by_id = {record.task_id: record for record in self.labels}
        if len(ranking_by_id) != len(self.rankings):
            raise ValueError("2Wiki ranking task IDs must be unique")
        if len(label_by_id) != len(self.labels):
            raise ValueError("2Wiki label task IDs must be unique")
        if set(ranking_by_id) != set(label_by_id):
            raise ValueError("2Wiki ranking and label task IDs must align")
        for task_id, label in label_by_id.items():
            valid = ranking_by_id[task_id].candidate_ids
            missing = set(label.gold_evidence_sentence_ids) - valid
            if missing:
                raise ValueError(
                    f"task_id={task_id} gold sentences do not exist: {sorted(missing)}"
                )
            for source, target in label.gold_dependency_edges:
                if source not in valid or target not in valid:
                    raise ValueError(
                        f"task_id={task_id} gold dependency edge references "
                        "a missing candidate"
                    )
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


@dataclass(frozen=True)
class ConvertedTwoWikiExample:
    ranking_record: TwoWikiRankingRecord
    label_record: TwoWikiLabelRecord


@dataclass(frozen=True)
class TwoWikiConversionResult:
    ranking_records: list[TwoWikiRankingRecord]
    label_records: list[TwoWikiLabelRecord]


__all__ = [
    "CombinedTwoWikiRecord",
    "ConvertedTwoWikiExample",
    "TwoWikiCandidateSentence",
    "TwoWikiConversionResult",
    "TwoWikiDocument",
    "TwoWikiEvidenceTriple",
    "TwoWikiExample",
    "TwoWikiLabelRecord",
    "TwoWikiPreparedSplit",
    "TwoWikiRankingRecord",
    "TwoWikiSupportingFact",
]
