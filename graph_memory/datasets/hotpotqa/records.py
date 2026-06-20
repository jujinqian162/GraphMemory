from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


class HotpotQACandidateSentence(TypedDict):
    sentence_id: str
    title: str
    sentence_index: int
    position: int
    text: str


class HotpotQARankingRecord(TypedDict):
    task_id: str
    question: str
    candidate_sentences: list[HotpotQACandidateSentence]


class HotpotQALabelRecord(TypedDict):
    task_id: str
    gold_answer: str
    gold_evidence_sentence_ids: list[str]
    gold_dependency_edges: list[list[str]]


class CombinedHotpotQARecord(HotpotQARankingRecord, HotpotQALabelRecord):
    """Combined HotpotQA inspection artifact; retrieval code must not consume it."""


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
