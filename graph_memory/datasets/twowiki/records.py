from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from graph_memory.contracts.common import JsonValue


class TwoWikiCandidateSentence(TypedDict):
    sentence_id: str
    title: str
    sentence_index: int
    position: int
    text: str


class TwoWikiRankingRecord(TypedDict):
    task_id: str
    question: str
    question_type: str
    candidate_sentences: list[TwoWikiCandidateSentence]
    metadata: dict[str, JsonValue]


class TwoWikiLabelRecord(TypedDict):
    task_id: str
    gold_answer: str
    gold_evidence_sentence_ids: list[str]
    gold_dependency_edges: list[list[str]]
    metadata: dict[str, JsonValue]


class CombinedTwoWikiRecord(TwoWikiRankingRecord, TwoWikiLabelRecord):
    """Combined 2Wiki inspection artifact; retrieval code must not consume it."""


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
