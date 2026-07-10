from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from graph_memory.contracts.common import JsonValue


class MuSiQueCandidateParagraph(TypedDict):
    paragraph_id: str
    title: str
    paragraph_index: int
    position: int
    text: str


class MuSiQueRankingRecord(TypedDict):
    task_id: str
    question: str
    candidate_paragraphs: list[MuSiQueCandidateParagraph]
    metadata: dict[str, JsonValue]


class MuSiQueLabelRecord(TypedDict):
    task_id: str
    gold_answer: str
    gold_answer_aliases: list[str]
    gold_evidence_paragraph_ids: list[str]
    gold_dependency_edges: list[list[str]]
    metadata: dict[str, JsonValue]


class CombinedMuSiQueRecord(MuSiQueRankingRecord, MuSiQueLabelRecord):
    """Combined MuSiQue inspection artifact; retrieval code must not consume it."""


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
